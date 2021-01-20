"""
Main module definitions in here
"""
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform


from freelunch import tech, zoo
from freelunch.base import continuous_space_optimiser
from freelunch.darwin import DE_methods, update_strategy_ps, select_strategy, adaptable_normal_parameter



class DE(continuous_space_optimiser):
    '''
    Differential evolution
    '''
    name = 'Differential Evolution'
    tags = ['continuous domain', 'population based', 'evolutionary']
    hyper_definitions = {
        'N':'Population size (int)',
        'G':'Number of generations (int)',
        'F':'Mutation parameter (float in [0,1])',
        'Cr':'Crossover probability (float in [0,1])'
    }
    hyper_defaults = {
        'N':100,
        'G':100,
        'F':0.5,
        'Cr':0.2
    }

    def run(self):
        pop = tech.uniform_continuous_init(self.bounds, self.hypers['N'])
        tech.compute_obj(pop, self.obj)
        for gen in range(self.hypers['G']):
            trial_pop = np.empty_like(pop, dtype=object)
            for i, sol in enumerate(pop):
                pts = np.random.choice(pop, 3)
                trial = zoo.animal()
                trial.dna = (pts[0].dna - pts[1].dna) + self.hypers['F'] * pts[2].dna
                trial.dna = tech.binary_crossover(sol.dna, trial.dna, self.hypers['Cr'])
                trial.dna = tech.apply_sticky_bounds(trial.dna, self.bounds)
                trial_pop[i] = trial
            tech.compute_obj(trial_pop, self.obj)
            pop = tech.sotf(pop, trial_pop)
        return pop

    
class SADE(continuous_space_optimiser):
    '''
    Self-Adapting Differential evolution
    '''
    name = 'Self-Adapting Differential Evolution'
    tags = ['continuous domain', 'population based', 'evolutionary', 'adaptive']
    hyper_definitions = {
        'N':'Population size (int)',
        'G':'Number of generations (int)',
        'F_u':'Mutation parameter initial mean (float in [0,2])',
        'F_sig':'Mutation parameter initial standard deviation (float in [0,1])',
        'Cr_u':'Crossover probability initial mean (float in [0,1])',
        'Cr_sig':'Crossover probability initial standard deviation (float in [0,1])',
        'Lp':'Learning period'
    }
    hyper_defaults = {
        'N':100,
        'G':100,
        'F_u':0.5,
        'F_sig':0.2,
        'Cr_u':0.5,
        'Cr_sig':0.2,
        'Lp':10
    }

    def run(self):
        #initial params and operations
        F = adaptable_normal_parameter(self.hypers['F_u'], self.hypers['F_sig'])
        Cr = adaptable_normal_parameter(self.hypers['Cr_u'], self.hypers['Cr_sig'])
        ops = np.array([o() for o in DE_methods.values()])
        #initial pop 
        pop = tech.uniform_continuous_init(self.bounds, self.hypers['N'])
        tech.compute_obj(pop, self.obj)
        #main loop
        for gen in range(self.hypers['G']):
            # parameter update
            if gen != 0 and gen%self.hypers['Lp']==0:
                update_strategy_ps(ops)
                F.update()
                Cr.update()
            #trial pop
            trial_pop = np.empty_like(pop, dtype=object)
            for i, sol in enumerate(pop):
                # mutation
                new = zoo.animal()
                op = select_strategy(ops)
                new.dna = op(sol, pop=pop, F=F())
                # crossover and apply bounds
                new.dna = tech.binary_crossover(sol.dna, new.dna, Cr())
                new.dna = tech.apply_sticky_bounds(new.dna, self.bounds)
                # Record adaptions
                new.tech.extend([op, F, Cr])
                trial_pop[i] = new
            # compute objectives
            tech.compute_obj(trial_pop, self.obj)
            # adaptive sotf
            pop = tech.sotf(pop, trial_pop)
        return pop

    

class SA(continuous_space_optimiser):
    '''
    Simulated Annealing 
    '''
    name = 'Simulated Annealing'
    tags = ['Continuous domain', 'Annealing']
    hyper_definitions = {
        'K':'number of timesteps (int)',
        'T':'Temperatures (np.ndarray, T.shape=(k,))',
        'P':'Acceptance probability (P(e, e, T) -> bool)'
    }

    def run(self):
        raise NotImplementedError


class PSO(continuous_space_optimiser):
    '''
    Base Class for Particle Swarm Optimisations
    '''

    name = 'Particle Swarm Optimisation'
    tags = ['Continuous domain', 'Particle swarm']
    hyper_definitions = {
        'N':'Population size (int)',
        'G':'Number of generations (int)',
        'I':'Inertia coefficients (np.array, I.shape=(2,))',
        'A':'Acceleration (np.array, I.shape=(2,))'
    }
    hyper_defaults = {
        'N':100,
        'G':200,
        'I':np.array([0.1, 0.9]),
        'A':np.array([0.1, 0.1])
    }

    def init_pop(self,N):
        # Function which initialises the swarm
        pop = np.empty((N,), dtype=object)
        for i in range(N):
            pop[i] = zoo.particle(np.array([np.random.uniform(a,b) for a, b in self.bounds]))
            pop[i].vel = np.squeeze((2*np.random.rand(self.bounds.shape[0],1)-1)*np.diff(self.bounds))
        return pop

    def move_swarm(self, pop, gen):
        # Basic particle swarm move

        inertia = self.hypers['I'][1]-(self.hypers['I'][1]-self.hypers['I'][0])*gen/self.hypers['G']
        # This loop is slow, should vectorise at some point
        for p in pop:
            p.vel = inertia*p.vel + \
                self.hypers['A'][0]*np.random.rand()*(p.best_pos-p.pos) + \
                self.hypers['A'][1]*np.random.rand()*(self.g_best.pos-p.pos)
            p.pos = tech.apply_sticky_bounds(p.pos + p.vel, self.bounds)
        return pop

    def test_pop(self, pop):
        # Test all population and update bests
        tech.compute_obj(pop, self.obj)
        return pop

    def best_particle(self, pop):
        # Particles particles on the wall, who's the bestest of them all
        best = zoo.particle()
        loc = np.squeeze(np.where(pop==min(pop,key=lambda x: x.best)))
        best.fitness = pop[loc].best
        best.pos = pop[loc].best_pos
        return best

    # Generic PSO run routine
    def run(self):

        # Initialise the swarm
        pop = self.init_pop(self.hypers['N'])

        # Test Initial Population
        self.test_pop(pop)
        self.g_best = self.best_particle(pop)

        # Main loop
        for gen in range(self.hypers['G']):

            # Propagate the swarm
            pop = self.move_swarm(pop,gen)

            # Test new swarm locations
            self.test_pop(pop)

            # Particle class updates best previous position
            # Update global best
            self.g_best = self.best_particle(pop)


        return sorted([ p.as_sol() for p in pop ],key=lambda x: x.fitness)

    
class KrillHerd(continuous_space_optimiser):
    '''
    Krill Herd Optimisation

    Krill move based on three things:
        1) Induced motion
        2) Foraging
        3) Random Physical Diffusion
    
    dX/dt = N + F + D

    Gandomi, Amir Hossein, and Amir Hossein Alavi. "Krill herd: a new bio-inspired optimization algorithm." Communications in nonlinear science and numerical simulation 17.12 (2012): 4831-4845.

    '''

    name = 'Krill Herd'
    tags = ['Continuous domain', 'Animal', 'Krill Herd']
    hyper_definitions = {
        'N':'Population size (int)',
        'G':'Number of generations (int)',
        'Ct': 'Control time step element in (0,2] (float64)',
        'Imotion':'Inertia coefficients for induced motion (np.array, I.shape=(2,))',
        'Iforage':'Inertia coefficients for foraging (np.array, I.shape=(2,))',
        'eps':'epsilon to stop div by 0 errors (small constant) (float64)',
        'Nmax': 'Maximum induced speed in the paper somewhat confusingly (float64)',
        'Vf':'Foraging speed (float64)',
        'Dmax':'Maximum diffusion speed in [0.002,0.010] (float64)',
        'Crossover':'Implement crossover (bool)',
        'Mutate':'Implement mutation (bool)',
        'Mu':'Mutation mixing parameter in (0,1) (float64)'
    }
    hyper_defaults = {
        'N':100,
        'G':250,
        'Ct':0.5, # NOTE: in the paper this is chosen as a random number in (0,2]
        'Imotion':np.array([0.9, 0.1]), 
        'Iforage':np.array([0.9, 0.1]), 
        'eps':1e-12,
        'Nmax':0.01, 
        'Vf':0.02, 
        'Dmax':0.005, # NOTE: in the paper this is chosen as a random number in [0.002,0.010]
        'Crossover':True,
        'Mutate':True,
        'Mu':0.5
    }

    def init_pop(self,N):
        # Function which initialises the krill randomly within the bounds 
        pop = np.empty((N,), dtype=object)
        for i in range(N):
            pop[i] = zoo.krill( \
                pos= np.array([np.random.uniform(a,b) for a, b in self.bounds]), \
                motion= 0.01*np.random.rand(1,self.bounds.shape[0]), \
                forage= 0.008*np.random.rand(1,self.bounds.shape[0]) + 0.002)
        return pop
        

    def get_herd(self,pop):
        '''
        It is more convenient to work with matrix representations of the krill
        '''

        vals = np.zeros(self.hypers['N'])
        locs = np.zeros((self.hypers['N'],self.obj.n))
        motion = np.zeros((self.hypers['N'],self.obj.n))
        forage = np.zeros((self.hypers['N'],self.obj.n))

        for i,krill in enumerate(pop):
            vals[i] = krill.fitness
            locs[i,:] = krill.pos
            motion[i,:] = krill.motion
            forage[i,:] = krill.forage

        return [vals,locs,motion,forage]



    def winners_and_losers(self,herd):
        '''
        Sometimes in life you're the best sometimes you're the worst, sorry krill
        '''
        win_idx = np.argmin(herd[0])
        lose_idx = np.argmax(herd[0])
        
        # Winner and loser are tuples of best/worst (fitness,location)
        winner = (herd[0][win_idx], herd[1][win_idx,:])
        loser = (herd[0][lose_idx], herd[1][lose_idx,:])

        return (winner,loser)

    def all_time_champion(self,pop):
        '''
        Best krill ever
        '''
        best = pop[0].best
        best_pos = pop[0].best_pos
        for p in pop[1:]:
            if p.best < best:
                best = p.best
                best_pos = p.best_pos

        return (best,best_pos)

    def local_motion(self,herd,gen):

        # pairwise distances between krill
        dists = squareform(pdist(herd[1]), 'euclidean')

        # Who's my neighbour
        sense_dist = np.sum(dists,axis=1)/5/self.hypers['N']
        neighbours = dists <= sense_dist

        winner, loser = self.winners_and_losers(herd)
        spread = loser[0] - winner[0]
        if spread == 0:
            raise tech.SolutionCollapseError

        # Alpha stores local [0] and target [1] for each krill 
        alpha = [np.zeros_like(herd[1]), np.zeros_like(herd[1])]

        # Alpha local, the effect of the neighbours
        for i in range(dists.shape[0]):
            Khat = (herd[0][i] - herd[0][neighbours[i,:]]) / spread
            Xhat = (herd[1][neighbours[i,:],:] - herd[1][i,:]) / (dists[i,neighbours[i,:]] + self.hypers['eps'])[:,None]
            alpha[0][i,:] = np.sum( Xhat * Khat[:,None] , axis=0)
            
        # Exploration/exploitation coefficient
        Cbest = 2*(np.random.rand() + gen / self.hypers['G'])

        # Alpha target, take me to your leader
        Kbest = (herd[0] - winner[0]) / spread
        Xbest = (winner[1] - herd[1]) / (dists[:,np.argmin(herd[0])][:,None] + self.hypers['eps'])
        alpha[1] = Cbest*Kbest[:,None]*Xbest

        # Alpha is weighted combination of local and target
        alpha = alpha[0] + alpha[1]

        inertia  = tech.lin_reduce(self.hypers['Imotion'],gen,self.hypers['G']) 
        return self.hypers['Nmax']*alpha + inertia*herd[2], Kbest
        

    def foraging(self,herd,gen,pop):

        winner, loser = self.winners_and_losers(herd)
        spread = loser[0] - winner[0]

        # Tasty food at the centre of mass but how good is it
        #Xfood = (np.sum(herd[1]/herd[0][:,None], axis=0) / np.sum(1/herd[0]))[:,None].T
        Xfood = np.average(herd[1],weights=1/herd[0],axis=0)
        Kfood = self.obj(Xfood)

        Xhat_food = (Xfood - herd[1]) / ( cdist( herd[1], Xfood[:,None].T, 'euclidean') + self.hypers['eps'] )
        Khat_food = (herd[0] - Kfood) / spread

        # Exploration/exploitation coefficient
        Cfood = 2*(1 - gen / self.hypers['G'])

        # Beta food
        beta_food = Cfood * Xhat_food * Khat_food[:,None]

        # Get best previous locations from population
        herd_best = [np.zeros_like(herd[0]), np.zeros_like(herd[1])]
        for i,krill in enumerate(pop):
            herd_best[0][i] = krill.best
            herd_best[1][i,:] = krill.best_pos

        # Xhat and Khat against best previous positions
        Xhat_best = (Xfood - herd_best[1]) / ( cdist( herd_best[1], Xfood[:,None].T, 'euclidean') + self.hypers['eps'] )
        Khat_best = (herd_best[0] - Kfood) / spread

        # Beta best
        beta_best = Khat_best[:,None]*Xhat_best
    
        beta = beta_food + beta_best

        # Foraging motion
        inertia = tech.lin_reduce(self.hypers['Iforage'],gen,self.hypers['G']) 
        return self.hypers['Vf']*beta + inertia*herd[3]


    def random_diffusion(self,gen):

        delta = 2*np.random.rand(self.hypers['N'],self.obj.n) - 1

        return self.hypers['Dmax']*( 1 - gen/self.hypers['G'])*delta
        


    def run(self):

        pop = self.init_pop(self.hypers['N'])

        # Compute first set of fitness
        pop = tech.compute_obj(pop, self.obj)
  

        # Determine time step as in paper
        if self.bounds is None:
            dt = 10 # If no bounds set use default
        else:
            bounds = tech.bounds_as_mat(self.bounds)
            dt = self.hypers['Ct']*np.sum(bounds[:,1]-bounds[:,0])


        # Main loop 
        for gen in range(self.hypers['G']):

            herd = self.get_herd(pop)

            # Induced motion, following the crowd
            N, Khat_best = self.local_motion(herd,gen)

            # Foraging motion
            F = self.foraging(herd,gen,pop)

            # Random diffusion
            D = self.random_diffusion(gen)

            # Total Velocity
            V = N + F + D

            # Genetic operations
            champion = self.all_time_champion(pop)

            # Crossover
            if self.hypers['Crossover']:
                crossover_prob = 0.2*Khat_best
                xover_herd = herd[1][np.random.randint(self.hypers['N'],size=(self.hypers['N'],)),:]
                for i,h in enumerate(herd[1]):
                    # Implement 1-to-gbest X-over not 1-to-rand as in paper...
                    xover_herd[i,:] = tech.binary_crossover(h,champion[1],crossover_prob[i])
                current_herd = xover_herd
            else:
                current_herd = herd[1]

                
            # Mutation
            if self.hypers['Mutate']:
                mutate_prob = 0.05/Khat_best
                mutate_prob[mutate_prob==np.Inf] = 0
                inds = np.random.randint(self.hypers['N'],size=(self.hypers['N'],2))
                mutates = np.random.rand(self.hypers['N'],self.obj.n) < mutate_prob[:,None]
                mutant_dna = champion[1] + self.hypers['Mu']*(herd[1][inds[:,0],:]-herd[1][inds[:,1],:])
                current_herd[mutates] = mutant_dna[mutates]

            # Move the herd
            new_pos = current_herd + dt*V

            # Compute objectives and update the herd
            for i,(dna,motion,forage) in enumerate(zip(new_pos,N,F)):
                pop[i].pos = dna
                pop[i].motion = motion
                pop[i].forage = forage
                pop[i].fitness = self.obj(dna)

        return sorted([ p.as_sol() for p in pop ],key=lambda x: x.fitness)

