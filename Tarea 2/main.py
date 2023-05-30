import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PatchCollection as plt_circles

# plt.rcParams['figure.dpi'] = 200
# plt.rcParams['savefig.dpi'] = 200



class Particles():

    particle_list = list()
    
    collision = False

    def __init__(self,r,m):
        Particles.particle_list.append(self)
        self.r = r
        self.A = np.pi*self.r**2
        self.s = self.A*4/np.pi
        
        self.m = m

        x = np.random.uniform(0,1)*self.L
        y = np.random.uniform(0,1)*self.L
        self.X = np.array([x,y])

        theta = np.random.uniform(0,2*np.pi)
        Vx = Particles.V0*np.cos(theta)
        Vy = Particles.V0*np.sin(theta)
        self.V = np.array([Vx,Vy])

        self.dp = 0
        

    def get_Energy(self):
        E = 0.5*self.m*np.linalg.norm(self.V)**2
        return E
        

    def step(self):

        self.update_wall()
        self.update_position()


    def update_position(self):        
        self.X += self.V*Particles.dt


    def update_collision(self,particle2):
        #print(self.get_Energy()+particle2.get_Energy())

        r = (self.X-particle2.X)/np.linalg.norm(self.X-particle2.X)
        q = -2*(self.m**2/(2*self.m))*(np.dot((self.V-particle2.V),r)*r)

        self.V += q/self.m
        particle2.V -= q/particle2.m


        #Particles.collision_number += 1
 
        #print(self.get_Energy()+particle2.get_Energy())

    def update_wall(self):
        flagx,flagy = self.check_wall()
        dp = 0
        if flagx:
            self.V[0] *= -1
            dp += 2*self.m*np.abs(self.V[0])
        if flagy:
            self.V[1] *= -1
            dp += 2*self.m*np.abs(self.V[0])
        self.dp = dp

    def check_wall(self):
        x,y = self.X
        ret = np.array([0,0])
        if (x+self.r >= self.L) or (x-self.r <= 0):
            ret[0] = 1
        if (y+self.r >=  self.L) or (y-self.r <= 0):
            ret[1] = 1
        return ret

    def check_collision(self,particle2):
        if np.linalg.norm(self.X - particle2.X) <= self.r + particle2.r:
            return True
        else:
            return False
        



class Simulation():

    def __init__(self, Particles, 
                 N=100, 
                 L = 1.0, 
                 V0 = 0.05,
                 r=0.02,
                 m=1):
        
        self.Particles = Particles
        self.N_particles = N
        self.Particles.L = L
        self.Particles.V0 = V0
        self.Particles.r = r
        self.Particles.m = m

        self.dt = 0.05 #0.9*self.Particles.r/self.Particles.V0
        self.Particles.dt = self.dt
        self.L = L


        self.total_dp = 0
        self.P = 0
        self.T = 0
        collision_number = 0


    def create_particles(self):
            
        print('Creating particles')
        for i in range(self.N_particles):
            flag = True
            while flag:
                flag = False
                particle = self.Particles(r=self.Particles.r, m=self.Particles.m)
                wall = particle.check_wall()
                s = wall.sum()
                if s>0:
                    flag = True
                    self.Particles.particle_list.remove(particle)
                    del particle
                else:
                    for particle2 in Particles.particle_list:
                        if particle is particle2:
                            continue
                        elif particle.check_collision(particle2):
                            flag = True
                            self.Particles.particle_list.remove(particle)
                            del particle
                            break
        print('particles created')



    def simulation_step(self):
        
        L_particles = list(self.Particles.particle_list)
        N_total = len(L_particles)

        for i in range(N_total):
            for j in range(i+1,N_total):
                particle1 = L_particles[i]
                particle2 = L_particles[j]
                if particle1.check_collision(particle2):
                    particle1.update_collision(particle2)
                    #particle2.update_collision(particle1)

            particle1.step()
                    
            self.total_dp += particle1.dp
            self.E += particle1.get_Energy()  

            if self.plot:
                self.x_particles.append(particle1.X[0])
                self.y_particles.append(particle1.X[1])
                self.circles.append(plt.Circle((particle1.X[0],particle1.X[1]),linewidth=0 ,radius=particle1.r, color='k'))                
        


    def update_variables(self):
        N = len(self.Particles.particle_list)
        self.P = self.total_dp/(self.Particles.dt*self.Particles.L*N)

        kb = 1.380649*10**-23
        self.T = (2/3)*(1/(N*kb))*self.E - 273.15 


    def run_simulation(self, N_steps=10, plot=False):
        
        self.create_particles()

        self.plot = plot
        if self.plot:         
            fig, self.ax = plt.subplots()
            self.ax.set_box_aspect(1)
            self.ax.set_xlim([0,Particles.L])
            self.ax.set_ylim([0,Particles.L])
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')
            anim = animation.FuncAnimation(fig, self.plot_particles, interval=1)
            plt.show()
        else:
            for n in range(N_steps):
                self.time_step()

    def time_step(self):
        self.total_dp = 0
        self.E = 0
        self.simulation_step()
        self.update_variables()      


    def plot_particles(self, i):
        self.circles = list()
        self.x_particles = list()
        self.y_particles = list()
        self.ax.cla()
        self.time_step()
        c = plt_circles(self.circles)
        self.ax.add_collection(c)
        plt.title(f'T={self.T}   P={self.P}')


########################################################################


IdealGas = Simulation(Particles, N=40)
IdealGas.run_simulation(plot=True)

 


