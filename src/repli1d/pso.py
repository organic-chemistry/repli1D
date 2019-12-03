#------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Simple Particle Swarm Optimization (PSO) with Python
#   July, 2016
#
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division
import random
import math

#--- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def func1(x):
    total=0
    for i in range(len(x)):
        total+=x[i]**2
    return total

#--- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self,x0,velocity_scale):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-velocity_scale,velocity_scale))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc,value=None):
        if value is None:
            self.err_i=costFunc(self.position_i)
        else:
            self.err_i = value

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        sum = 0
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[1]:
                self.position_i[i]=bounds[1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[0]:
                self.position_i[i]=bounds[0]

            sum += self.position_i[i]

        #Normalise
        for i in range(0,num_dimensions):
            self.position_i[i] /= sum




class PSO():
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter,velocity_scale,normed=False):
        global num_dimensions

        print("Parameters",bounds,num_particles,maxiter,velocity_scale)
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        if normed:
            assert(abs(sum(x0)-1) < 1e-3)

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0,velocity_scale=velocity_scale))
        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):

                if i == 0 and j > 0:
                    #No need to evaluate several time the first position
                    #print("No comupte")
                    swarm[j].evaluate(costFunc,value=swarm[0].err_i)
                else:
                    #print("Compute")
                    swarm[j].evaluate(costFunc)
                    #print(swarm[j].err_i)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

            print ("Iteration i=%i, %.4f"%(i,err_best_g))

        # print final results
        print ('FINAL:')
        #print (pos_best_g)
        print (err_best_g)
        self.best = pos_best_g



#--- RUN ----------------------------------------------------------------------+
