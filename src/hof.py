class HOF():
    def __init__(self, capacity, fitness):
        self.minds = []
        self.values = []
        self.capacity = capacity
        self.fitness = fitness
    def fill_up(self, threshold, arena, l):
        i = 0
        while i < self.capacity:

            arena.ant.brain = Mind(arena.ant.brain.N_minds, arena.ant.brain.hidden_sizes, arena.ant.brain.device)
            brain = arena.ant.brain
            arena.reinit()
            s = t.stack(arena.raw_run(l)).refine_names("T", "B")
            result = self.fitness(s).unflatten("B", (("M", arena.ant.brain.N_minds),("B", s.size("B")//arena.ant.brain.N_minds))).mean("B")
            for k, score in enumerate(result):
                if score >  threshold:
                    self.minds.append(brain.resized_copy(k, 1))
                    self.values.append(score)
                    i += 1
                    printProgressBar(i, self.capacity, f"{score.item():.4f}")
    def save(self, fname):
        lista = []
        for mind, v in zip(hof.minds, hof.values):
            lista.append((mind.dump(), v))

        t.save(lista, fname)

    def load(self, fname):
        lista = t.load(fname)
        for mindc, v in lista:
            print(v)
            mind = Mind(1)
            mind.load_from_dump(mindc)
            self.minds.append(mind)
            self.values.append(v)

    def pick_a_mind(self):
        chosen = self.minds[t.tensor(self.values).argsort()[-1]]
        return chosen
    def evolver(self, arena, N, length, N_minds):
        maxes = [] 
        chosen = self.pick_a_mind()
        new = chosen.resized_copy(0, N_minds )
        arena.ant.brain = new
        new.mutate(relative_d_mutate, relative_d_mutate)
        for i in range(N):
            arena.reinit()
            result = t.stack(arena.raw_run(length)).refine_names("T", "B")
            result = self.fitness(result)
            result = result.unflatten("B", (("M", arena.ant.brain.N_minds),("B", result.size("B")//arena.ant.brain.N_minds))).mean("B")
        
            maxi = result.max("M")[0]
            maxes.append(maxi.cpu().item())
            arena.ant.brain.mutate(*extendmutator(result)) 
            arena.ant.brain.mutate(relative_d_mutate, relative_d_mutate)
            printProgressBar(i, N, i, f"max food: {maxi.item()}")
        return maxes
