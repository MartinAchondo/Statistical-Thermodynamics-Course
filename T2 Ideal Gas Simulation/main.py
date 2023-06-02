import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PatchCollection as plt_circles
from PIL import Image, ImageDraw
from collections import deque
from tqdm import tqdm as log_progress
import os
from scipy.stats import maxwell

# plt.rcParams['figure.dpi'] = 200
# plt.rcParams['savefig.dpi'] = 200



class Particles():

    particle_list = list()
    

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
        
    @property
    def E(self):
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
            dp += 2*self.m*np.abs(self.V[1])
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
                 m=1,
                 dt=0.01):
        
        self.Particles = Particles
        self.N_particles = N
        self.Particles.L = L
        self.Particles.V0 = V0
        self.Particles.r = r
        self.Particles.m = m

        self.dt = dt#0.9*self.Particles.r/self.Particles.V0
        self.Particles.dt = self.dt
        self.L = L


        self.total_dp = 0
        self.P = 0
        self.T = 0

        self.kb = 1.380649*10**-23
        self.collision_number = 0
        self.L_P = deque([0]*10)
        self.L_T = deque([0]*10)

        self.L_residual = list()

        self.dir_path = 'results'


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
                    self.collision_number += 1
                    #particle2.update_collision(particle1)

            particle1.step()
                    
            self.total_dp += particle1.dp
            self.E += particle1.E  


    def update_variables(self):
        N = self.N_particles
        P = self.total_dp/(self.Particles.dt*self.Particles.L*N)
        T = (2/2)*(1/(N*self.kb))*self.E

        self.L_P.append(P)
        self.L_T.append(T)
        self.P = np.array(self.L_P).mean()
        self.T = np.array(self.L_T).mean()

        self.residual = np.abs(self.P*self.L**2 - N*self.kb*self.T)
        self.L_residual.append(self.residual)


    def run_simulation(self, N_steps=10, plot=False):
        
        self.N_steps = N_steps
        self.plot = plot

        self.create_particles()

        frames = list()

        pbar = log_progress(range(self.N_steps))
        pbar.set_description("Residual: %s " % 1000)
        for n in pbar:
            if self.plot:
                if n%1==0:
                    frame = self.create_frame_image()
                    frames.append(frame)
            self.time_step(n)
            if n % 10 == 0:
                pbar.set_description("Residual: {:6.4e}, Collisions: {}".format(self.residual,self.collision_number))           
        
        if self.plot:
            self.save_animation(frames)


    def time_step(self,n):
        self.L_P.popleft()
        self.L_T.popleft()
        
        self.total_dp = 0
        self.E = 0
        self.simulation_step()
        self.update_variables()


    def postprocessing(self):

        print(f'Pressure = {self.P} [Pa]')
        print(f'Temperature = {self.T} [K]') 
        print(f'Residual = {self.residual}')
        print(f'Total collisions: {self.collision_number}')
        print(f'Final time {self.dt*self.N_steps} [s]')

        self.plot_residual()
        self.plot_velocity()
        self.plot_particles()


    def plot_residual(self):
        fig, ax = plt.subplots()
        ax.plot(self.L_residual, label='Residual ', c='b')
        ax.set_xlabel('n')
        ax.set_ylabel(r'$|PV-Nk_{b}T|$')
        #ax.set_title(f'Solution {text_l} of PDE, Iterations: {self.NN.N_iters}, Loss: {loss}')
        ax.grid()
        ax.legend()
        name = 'Residual.png'
        fig.savefig(os.path.join(self.dir_path,name))


    def plot_velocity(self):
        fig, ax = plt.subplots()
        L_V = list()
        for particle in self.Particles.particle_list:
            V = np.sqrt(particle.V[0]**2 + particle.V[1]**2)
            L_V.append(V)
        sns.histplot(L_V, stat='density', label='Simulation', color='#9dbbeb', edgecolor='#e9f2f1')
        params = maxwell.fit(L_V)

        x = np.linspace(0, np.max(L_V), 100)
        y = maxwell.pdf(x, *params)
        sns.lineplot(x=x, y=y, color='b', label='Fitted Distribution', ax=ax)


        ax.grid(True)
        ax.set_xlabel('Velocity')
        ax.set_ylabel('Density')
        #ax.set_ylim(bottom=0)
        #ax.set_xlim(bottom=0)

        ax.legend()

        name = 'Velocity_distribution.png'
        fig.savefig(os.path.join(self.dir_path,name))


    def plot_particles(self):

        fig, ax = plt.subplots()

        self.circles = list()
        self.x_particles = list()
        self.y_particles = list()

        for particle in self.Particles.particle_list:
            self.x_particles.append(particle.X[0])
            self.y_particles.append(particle.X[1])
            self.circles.append(plt.Circle((particle.X[0],particle.X[1]),linewidth=0 ,radius=particle.r, color='k'))    

        c = plt_circles(self.circles)
        ax.add_collection(c)
        plt.title(f'Step = {self.N_steps}')
      
        ax.set_box_aspect(1)
        ax.set_xlim([0,Particles.L])
        ax.set_ylim([0,Particles.L])
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        name = 'Particles.png'
        fig.savefig(os.path.join(self.dir_path,name))




    def create_frame_image(self):
        scale_factor = 120 
        image_width = int(self.L * scale_factor)
        image_height = int(self.L * scale_factor)
        image = Image.new('RGB', (image_width, image_height), 'white')
        draw = ImageDraw.Draw(image)
        for particle in self.Particles.particle_list:
            x1 = int((particle.X[0] - particle.r) * scale_factor)
            y1 = int((particle.X[1] - particle.r) * scale_factor)
            x2 = int((particle.X[0] + particle.r) * scale_factor)
            y2 = int((particle.X[1] + particle.r) * scale_factor)
            draw.ellipse([(x1, y1), (x2, y2)], fill='blue')
        return image

    def save_animation(self, frames):
        name = 'gas_simulation.gif'
        frames[0].save(os.path.join(self.dir_path,name), save_all=True, append_images=frames[1:], optimize=False, duration=100, loop=0)


########################################################################


if __name__=='__main__':

    parameters = {
        'm': 1,  
        'r': 1,  
        'V0': 6 ,
        'L': 100,
        'dt': 0.005
    }

    IdealGas = Simulation(Particles, N=200, **parameters)
    IdealGas.run_simulation(N_steps=4000, plot=False)
    IdealGas.postprocessing()


 


