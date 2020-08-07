from matplotlib import pyplot as plt
import torch as t
import numpy as np

import sys
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    # Print New Line on Complete
    
class Ant():
    def __init__(self, v, N_b, N_start, phero_dens, turn_obst=1.0, turn_pher=1.0, lookforward=3):
      #  eredeti
        device=phero_dens.device
        self.turn_obst=turn_obst
        self.turn_pher=turn_pher
        self.lookforward = lookforward
        self.v = v
        self.phis = t.rand(N_b, N_start, device=device).refine_names("B", "EX")*np.pi*2
        self.xs = t.zeros(N_b, N_start, device=device).refine_names("B", "EX") - 25.0
        self.ys = t.zeros(N_b, N_start, device=device).refine_names("B", "EX") - 25.0
        self.food_carry = t.zeros(N_b, N_start, device=device).refine_names("B", "EX")
        self.food_accum = t.zeros(N_b, device=device).refine_names("B")
        self.phero_dens = t.ones_like(self.phis).align_to("B", "EX", "PH", "ST")*phero_dens  #mennyi feromont rak le fajtánként
        self.turn_prob = t.ones_like(self.phis)*0.01
        
    def reinit(self):
        self.phis = t.rand_like(self.phis)*np.pi*2
        self.xs *= 0
        self.ys *= 0
        self.xs += -25
        self.ys += -25
        self.food_carry *= 0
        self.food_accum *= 0
        
    def step(self, dt, arena):
        self.xs += dt*self.v*t.cos(self.phis)
        self.ys += dt*self.v*t.sin(self.phis)
        self.f_p = arena.get_index(arena.colomap, self.xs, self.ys) #megnézi, hogy a hangyák melyik mezőkön állnak

    def render(self, background, arena):
         
        tens = t.zeros_like(background.narrow(-1, 0, 1).squeeze(-1))
        result = arena.update_field(tens, t.ones_like(self.food_carry), self.f_p).align_as(background)
        foodhaver = arena.update_field(tens, t.ones_like(self.food_carry)*self.food_carry, self.f_p).align_as(background)
        background *= ~(result > 0)
        background += t.cat((result-0.8*foodhaver, result, result-0.8*foodhaver), "C")
        return t.clamp(background, 0,1)

    def control(self, arena, dt):
     
            #kajagyűjtés
            foodat=arena.get_food(self.f_p)*(~(self.food_carry > 0))
            self.food_carry+=foodat
      #      self.phis += ( np.pi + t.atan2(self.ys,self.xs) - self.phis)*(foodat > 0)
          #  arena.update_food(-foodat,self.f_p)

            #kajalerakás
            based = arena.get_base(self.f_p)
            self.food_accum += (based*self.food_carry).sum("EX")
            self.food_carry *= ~(based > 0)
            fooded = (self.food_carry> 0)
            unfooded = ~fooded
      
            #feromonok
            phero = self.phero_dens*t.stack((fooded.rename(None), unfooded.rename(None)), -1).refine_names("B", "EX", "ST").align_as(self.phero_dens)
            arena.update_pheros(phero.sum("ST"), self.f_p)

            xleft,yleft = self.xs + self.lookforward*self.v*t.cos(self.phis+0.5),self.ys + self.lookforward*self.v*t.sin(self.phis+0.5)
            xright,yright = self.xs + self.lookforward*self.v*t.cos(self.phis-0.5),self.ys + self.lookforward*self.v*t.sin(self.phis-0.5)
            f_l, f_r = arena.get_index(arena.colomap, xleft, yleft), arena.get_index(arena.colomap, xright, yright)
            phero_left, phero_right = arena.get_pheros(f_l), arena.get_pheros(f_r)
            dpher = phero_left-phero_right
            self.phis+= (( dpher[:,:,0])*unfooded +(dpher[:,:,1])*fooded)*self.turn_pher

            #ez a random fordulás+bázis fele ha van kaja
            turn_prob = self.turn_prob*(t.exp(-(phero_left + phero_right)[:,:,0]*unfooded -(phero_left + phero_right)[:,:,1]*fooded))#+ fooded)
            turn = t.distributions.bernoulli.Bernoulli(turn_prob.rename(None)*dt).sample()
            amount = t.randn_like(turn)#*(unfooded) + ( np.pi + t.atan2(self.ys,self.xs) - self.phis)*fooded
            self.phis += 1.0*turn*amount

            #akadálykikerülés
            xleft,yleft=self.xs + self.lookforward*self.v*t.cos(self.phis+0.5),self.ys + self.lookforward*self.v*t.sin(self.phis+0.5)
            xright,yright=self.xs + self.lookforward*self.v*t.cos(self.phis-0.5),self.ys + self.lookforward*self.v*t.sin(self.phis-0.5)
            f_l, f_r = arena.get_index(arena.colomap, xleft, yleft), arena.get_index(arena.colomap, xright, yright)
            obs_left, obs_right = arena.get_obst(f_l), arena.get_obst(f_r)
            self.phis += -obs_left*self.turn_obst*dt + obs_right*self.turn_obst*dt + obs_left*obs_right*np.pi

class EvolvedAnt(Ant):
    def __init__(self, v, N_b, N_start, N_repeat, phero_dens, hidden_sizes = [32, 5], turn_obst=1.0, lookforward=3):
        super().__init__( v, N_b, N_start, phero_dens, turn_obst=turn_obst, turn_pher=1.0, lookforward=lookforward)
        self.brain=Mind(N_b, N_repeat,hidden_sizes, phero_dens.device)
    def __init__(self, v, N_b, N_start, brain, phero_dens, turn_obst=1.0, lookforward=3):
        super().__init__( v, N_b, N_start, phero_dens, turn_obst=turn_obst, turn_pher=1.0, lookforward=lookforward)
        self.brain=brain
    def control(self, arena, dt):
     
            #kajagyűjtés
            foodat=arena.get_food(self.f_p)*(~(self.food_carry > 0))
            self.food_carry+=foodat
      #      self.phis += ( np.pi + t.atan2(self.ys,self.xs) - self.phis)*(foodat > 0)
          #  arena.update_food(-foodat,self.f_p)

            #kajalerakás
            based = arena.get_base(self.f_p)
            self.food_accum += (based*self.food_carry).sum("EX")
            self.food_carry *= ~(based > 0)
            fooded = (self.food_carry> 0)
            unfooded = ~fooded
      
            #feromonok
           
            xleft,yleft = self.xs + self.lookforward*self.v*t.cos(self.phis+0.5),self.ys + self.lookforward*self.v*t.sin(self.phis+0.5)
            xright,yright = self.xs + self.lookforward*self.v*t.cos(self.phis-0.5),self.ys + self.lookforward*self.v*t.sin(self.phis-0.5)
            f_l, f_r = arena.get_index(arena.colomap, xleft, yleft), arena.get_index(arena.colomap, xright, yright)
            phero_left, phero_right = arena.get_pheros(f_l), arena.get_pheros(f_r)
            in_ = t.cat((phero_left, phero_right, self.food_carry.align_as(phero_left), t.randn_like(self.food_carry).align_as(phero_left)), -1).rename("B", "EX", "IN")
  
            out_ =self.brain.process(in_)
            self.phis += t.clamp(out_[:, :, 0], -0.5, 0.5)*dt
            arena.update_pheros(t.clamp(out_[:,:, 1:].rename("B", "EX", "PH"), 0, 0.1)*dt, self.f_p)

            #akadálykikerülés
            xleft,yleft=self.xs + self.lookforward*self.v*t.cos(self.phis+0.5),self.ys + self.lookforward*self.v*t.sin(self.phis+0.5)
            xright,yright=self.xs + self.lookforward*self.v*t.cos(self.phis-0.5),self.ys + self.lookforward*self.v*t.sin(self.phis-0.5)
            f_l, f_r = arena.get_index(arena.colomap, xleft, yleft), arena.get_index(arena.colomap, xright, yright)
            obs_left, obs_right = arena.get_obst(f_l), arena.get_obst(f_r)
            self.phis += -obs_left*self.turn_obst*dt + obs_right*self.turn_obst*dt + obs_left*obs_right*np.pi

        
class Mind():
      def __init__(self, N_minds, hidden_sizes = [32, 5], device="cpu"):
        self.Ws = []
        self.bs = []
        prev_size = 10
        self.hidden_sizes = hidden_sizes
        self.N_minds = N_minds
        self.device=device
        for s in hidden_sizes:
            self.Ws.append(t.randn(self.N_minds, s, prev_size,device= device).refine_names("M","OUT","IN").align_to("M","B","OUT","IN"))
            self.bs.append(t.randn(self.N_minds, s ,device=device).refine_names("M","OUT").align_to("M","B","EX","OUT"))
            prev_size = s

      def process(self,in_):
          N_repeat = in_.size("B")//self.N_minds
          in_ =  in_.align_to("B", "EX", "IN", "Q").unflatten("B", (("M", self.N_minds), ("B", N_repeat)))
          for W, b in zip(self.Ws, self.bs):
              out = t.matmul(W.align_to("M", "B","EX", "OUT", "IN"), t.relu(in_)).squeeze("Q") + b
              in_ = out.unflatten("OUT", (("IN", out.size("OUT")), ("Q", 1)) )
          return in_.squeeze("Q").flatten(("M", "B"), "B")

      def mutate(self, Wmutator, bmutator):
      
          for W, b in zip(self.Ws,self.bs): 
              nw = W.names
              nb = b.names
              W.rename_(None)
              b.rename_(None)
           
              W[:,:,:,:] = Wmutator(W)
              b[:,:,:,:] = bmutator(b)
              W.rename_(*nw)
              b.rename_(*nb)

      def resized_copy(self, i, n, op = lambda x, i: x[i]):
          smind = Mind(n, self.hidden_sizes, self.device)
          for W, b, sW, sb in zip(self.Ws, self.bs, smind.Ws, smind.bs):
              nw = sW.names
              nb = sb.names
              sW.rename_(None)
              sb.rename_(None)
              sW[:,:,:,:] = op(W.rename(None), i)
              sb[:,:,:,:] = op(b.rename(None), i)
              sW.rename_(*nw)
              sb.rename_(*nb)

          return smind

      def dump(self):
         dename = lambda l: list(map(lambda x: x.rename(None), l))
         return (dename(self.Ws), dename(self.bs), self.hidden_sizes, self.N_minds, self.device)

      def load_from_dump(self, seried):
          self.Ws, self.bs, self.hidden_sizes, self.N_minds, self.device = seried

          self.Ws = list(map(lambda x: x.refine_names("M","B","OUT","IN"), self.Ws))
          self.bs = list(map(lambda x: x.refine_names("M","B","EX","OUT"), self.bs))

import cv2

class Arena():
    def __init__(self, obstacle_field, food_field, pherocol, dt, ant, pherokern):
        N_pheromons = pherocol.size("PH")
        assert obstacle_field.shape == food_field.shape
        self.obstacle_field = obstacle_field.refine_names("B", "X", "Y")
        self.food_field = food_field.refine_names("B", "X", "Y")
        self.N_pheromons = N_pheromons
        self.pheromon_field = t.zeros(*obstacle_field.shape, N_pheromons, device=obstacle_field.device, names = ("B", "X", "Y", "PH"))
        self.colomap = t.zeros_like(self.obstacle_field)
        base_size = 4
        self.colomap[:, self.colomap.size(1)//4-base_size:self.colomap.size(1)//4+base_size, self.colomap.size(2)//4-base_size:self.colomap.size(2)//4+base_size] = 1.0
        self.dt = dt
        self.ant = ant
        self.pherocol = pherocol
        self.pherokern = pherokern
        self.s0 = t.cuda.Stream()
        self.s1 = t.cuda.Stream()
    def get_index(self, field, x, y):
        sx, sy = self.obstacle_field.size("X"), self.obstacle_field.size("Y")
        x_ind, y_ind = t.clamp(x.to(t.int64) + sx//2, 0, sx-1), t.clamp(y.to(t.int64) + sy//2, 0, sy-1)
        badd = t.arange(field.size("B"), device=field.device).refine_names("B")*sx*sy
        xadd = x_ind*sx
        f = (badd.align_as(y_ind) + xadd + y_ind )
        return f
    def get_field(self, field, f):
       
      
        res = t.index_select(field.flatten(("B", "X", "Y"), "All").rename(None), dim=-1, index=f.flatten(("B", "EX"), "All").rename(None)).view(f.shape).refine_names(*f.names)
        return res

    def update_field(self, field, delta, f):
       
        resc = field.flatten(("B", "X", "Y"), "All").rename(None)
        return resc.index_add(0, f.flatten(("B", "EX"), "All").rename(None), delta.flatten(("B", "EX"), "All").rename(None)).view(field.shape).refine_names(*field.names)

    def reinit(self):
      self.ant.reinit()
      self.pheromon_field*=0
      #self.food_field[:, 10:15, 90:95] = 1.0

    def maptens(self):
        pheromap = t.matmul(self.pherocol, self.pheromon_field.align_to("B", "X", "Y", "PH", "Q")).squeeze("Q")
       # pheromap = pheromap/pheromap.max("X", keepdim=True)[0].max("Y", keepdim=True)[0]
        base = t.stack((self.colomap.rename(None), self.food_field.rename(None), self.obstacle_field.rename(None)), -1).refine_names("B", "X", "Y", "C")
        return base + ~(base.sum("C", keepdim=True) > 0)*pheromap#.transpose(-2, -3).flip(-2)
    def plotup(self, i, ax):
        s1, s2 = self.obstacle_field.size("X"), self.obstacle_field.size("Y")
        return ax.imshow(self.maptens()[i], extent=(-s1//2, s1//2, -s1//2, s1//2))
    
    def step(self):
        
        self.ant.step(self.dt, self)
        
        self.pheromon_field += t.nn.functional.conv2d(
        self.pheromon_field.align_to("B", "PH", "X", "Y").rename(None), self.pherokern, padding=((self.pherokern.size(-2)-1)//2, (self.pherokern.size(-1)-1)//2)).refine_names("B", "PH", "X", "Y").align_to("B", "X", "Y", "PH")*self.dt
        self.pheromon_field = t.clamp(self.pheromon_field, 0, np.inf)
        self.ant.control(arena, self.dt)

    def get_food(self, f):
        return self.get_field(self.food_field, f)
    def update_food(self, delta, f):
        self.food_field = self.update_field(self.food_field, delta, f)
    def get_obst(self, f):
        return self.get_field(self.obstacle_field, f)
    def get_base(self, f):
        return self.get_field(self.colomap, f)
    def get_pheros(self, f):
      
        res = t.index_select(self.pheromon_field.flatten(("B", "X", "Y"), "All").rename(None), dim=0, 
                             index=f.flatten(("B", "EX"), "All").rename(None)).view(*f.shape, self.N_pheromons).refine_names(*f.names, "PH")
        return res
    def update_pheros(self, delta, f):

        self.pheromon_field = self.update_field(self.pheromon_field, delta, f)
        
    def videup2(self, N, u=4, i = 0):
        from threading import Thread
        from queue import Queue

        s1, s2 = self.obstacle_field.size("X"), self.obstacle_field.size("Y")
        writer = cv2.VideoWriter("output.avi",
                      
        cv2.VideoWriter_fourcc(*"XVID"), 30,(u*s2,u*s1))
        accumed = []
        q = Queue()
        def frametrans(tens):
            mp = cv2.normalize(tens.numpy(), None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return cv2.resize(mp, (s2*u,s1*u) , interpolation=0)

        def consumer():
            while True:
                res = q.get()
                res = frametrans(res.cpu())
                writer.write(res)
                q.task_done()
   
        th = Thread(target=consumer)
        th.daemon = True
        th.start()
    
        for frame in range(N):
            self.step()
            m = self.maptens()
            res = self.ant.render( m, self)
            q.put(res[i].cpu())
            accumed.append(self.ant.food_accum.clone().rename(None))
            self.s0.synchronize()
            if frame % 100 == 0:
                printProgressBar(frame, N)
        q.join()
        writer.release()
        return accumed
    def raw_run(self, N):
        accumed = []
        for frame in range(N):
            self.step()
            accumed.append(self.ant.food_accum.clone().rename(None))
        return accumed
