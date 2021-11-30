import numpy as np
import copy
#np.random.seed(0)
nc = 0
#import pylab


def comupute_enhanced(proba, actual_pos,cascade={}):

    #assert()

    infl = cascade["infl"] #,50)
    amount = cascade["amount"] #,4)
    down = cascade["down"]#,10)
    damount = cascade["damount"]#,1 / 10000.)

    cproba = np.zeros_like(proba) + 1
    for position, direction, init_position in actual_pos:
        # print(position,direction,cproba,max(position-infl),position)
        if direction == "L":
            cproba[position:min(position + infl, len(cproba) - 1)] *= amount
        if direction == "R":
            cproba[max(position - infl, 0):position] *= amount

        if direction == "L":
            cproba[position:min(position + down, len(cproba) - 1)] *= damount
        if direction == "R":
            cproba[max(position - down, 0):position] *= damount

    return cproba


def generate_newp_cascade(pos, proba, avail,actual_pos=[],cascade={}):
    #print("cascade",actual_pos)
    newp = []
    finished = False


    cproba = comupute_enhanced(proba, actual_pos,cascade)
    filter = proba != 0

    npossible = np.sum(filter)

    """
    f = pylab.figure()
    f.add_subplot(211)
    pylab.plot(cproba)
    f.add_subplot(212)
    pylab.plot(proba * cproba)
    pylab.plot(proba)

    pylab.show()
    """
    if npossible > avail:
        newp = list(np.random.choice(pos[filter], size=avail,replace=False,
                                p=proba[filter] * cproba[filter] / (np.sum(cproba[filter] * proba[filter]))))

    else:
        newp = list(pos[filter])
        finished = True

    for p in newp:
        proba[p] = 0

    newp.sort()

    return pos, proba, newp, finished


def generate_newp_no_corre(pos, proba, avail,actual_pos=[],cascade={},previous =[]):

    #Return a list of #avail site
    #Modify proba and previous
    #print(cascade)
    if cascade != {} :
        pos, proba, newp, finished = generate_newp_cascade(pos, proba, avail, actual_pos=actual_pos,cascade=cascade)
        return pos, proba, newp, finished,[]
    newp = []
    finished = False

    #first call generate ordered list of origins
    if previous == [] and np.sum(proba) != 0:
        size = np.sum(proba != 0)
        previous = list(np.random.choice(pos,size=size,
                                         replace=False,
                                          p=proba/np.sum(proba)))

    n = len(previous)

    # check for passivation
    for i in range(n)[::-1]:
        if proba[previous[i]] == 0:
            previous.pop(i)

    # generate avail position
    for p in range(n):
        if len(previous) != 0:
            next_pos = previous.pop(0)
            if proba[next_pos] != 0:
                newp.append(next_pos)
                proba[next_pos] = 0
            if len(newp) == avail:
                break
        else:
            finished = True

    # Set the proba of initiation to 0
    for ipos in newp:
        proba[ipos] = 0

    newp.sort()

    return pos, proba, newp, finished,previous



class Chrom:
    def __init__(self,start,end,timespend=[]):
        self.start = start
        self.end = end
        self.actual_pos = []  # Positions of the forks
        self.list_events = []
        self.delta = []
        self.rfd = np.zeros(end-start)
        self.mrt = np.zeros(end-start) + np.nan
        self.debug = False
        self.lt = []
        self.timespend = timespend
        self.t=0
        self.la = []
        self.oldp =[]
        #print("Times.",self.timespend)
        if len(self.timespend) != 0:
            # timespend is the item in kb necessary to travel a bin.
            # it must be an integer
            self.constant_speed = False

            assert(self.timespend.dtype==np.int)
        else:
            self.constant_speed = True
            #self.timespend = np.ones(end-start,dtype=np.int)

    def append_forks(self,pos):

        deb=False
        if deb:
            print("Append frks",pos)
            print("Before",self.actual_pos)


        if not np.isnan(self.mrt[self.rc(pos)]) :
            self.check(show=True,msg="Non null position %.2f, value %.2f" %(pos,self.mrt[self.rc(pos)]))
            raise "Non nul"


        self.mrt[self.rc(pos)]=self.t
        self.la.append(pos)
        if self.debug:
            print("New Fork",pos)

        shift=0
        if not self.constant_speed :
            shift=0.5
        found=False
        self.list_evens = []
        #print(self.actual_pos)
        if len(self.actual_pos) != 0 and self.actual_pos[0][0] > pos:
            if self.list_events != []:
                self.list_events[0] = [self.actual_pos[0][0]-pos,[1,2]]
                self.list_events.insert(0,[pos,[0]])
                #for p in self.list_events[2:]:


            self.actual_pos.insert(0, [pos+shift, "L",pos+shift])
            self.actual_pos.insert(1, [pos+shift, "R",pos+shift])
            found = True
        else:
            for p2 in range(len(self.actual_pos)-1):
                if self.actual_pos[p2+1][0] > pos >= self.actual_pos[p2][0]:
                    self.actual_pos.insert(p2+1, [pos+shift, "L",pos+shift])
                    self.actual_pos.insert(p2+2, [pos+shift, "R",pos+shift])
                    found = True
                    break

        if not found:
            self.actual_pos.extend([[pos+shift, "L",pos+shift], [pos+shift, "R",pos+shift]])


        if self.list_events == []:
            self.list_events.append([pos-self.start,[0]])
            self.list_events.append([self.end-pos,[1]])
        #print(self.actual_pos)
        #self.check()
        if deb:
            print("Afetr")
            print(self.actual_pos)

        self.list_events_comp()


    def list_events_comp(self,shift=0):
        # Compute the lists of next collisions
        if self.actual_pos == []:
            self.list_events = []
            return []
        delta = []
        pause = []
        start = 0
        end = None
        deb=False
        if deb:
            print("Startc")

        def time_to_meet(start=0,end=None,extremity=False):
            if self.constant_speed:
                time = end-start
            else:
                time = np.sum(self.timespend[self.rc(start):self.rc(end)])
                if self.rc(end)< len(self.timespend) - 1:
                    time += (round(end,5)-int(end)) * self.timespend[self.rc(end)]
                time -= (round(start,5) - int(start)) * self.timespend[self.rc(start)]
                #print(self.speed[start:end])
                #print("t",start,end,time,end-start,extremity)

            #only one fork progressing

            #if time > 100:
            #    print(start,end,time,extremity)

            if extremity:
                if self.constant_speed:
                    return int(time)
                else:
                    return time
            #Must be divided by 2 as both strand progress

            #print("h",start,end,time)
            if self.constant_speed:
                if time % 2 == 0:
                    return int(time // 2)
                else:
                    return int(time // 2 + 1)
            else:
                return time / 2


        if self.actual_pos[0][1] == "L":
            delta.append([time_to_meet(self.start,self.actual_pos[0][0],extremity=True), [0+shift]])

            if self.actual_pos[0][0]-self.start < 0:
                raise
            start = 1
            if deb:
                print("start,shift")
        if self.actual_pos[-1][1] == "R":
            end = -1

        if deb:
            print(start,end)
            print("Full",self.actual_pos)
            print("view",self.actual_pos[start:end])

        view = self.actual_pos[start:end]
        i = 0
        # if not (len(delta) == 2 and len(actual_pos) == 2):
        for p1, p2 in zip(view[::2], view[1::2]):
            """
            deltapt = p2[0]-p1[0]
            if deltapt % 2 == 0:
                deltapt = (p2[0]-p1[0])//2
            else:
                deltapt = (p2[0] - p1[0]) // 2 + 1
            """
            assert(p1[1] == "R")
            assert(p2[1] == "L")


            delta.append([time_to_meet(p1[0],p2[0]), [2*i+start+shift, 2*i+1+start+shift]])
            #print(p2[0] ,p1[0],(p2[0]-p1[0])//2+1)
            #print(delta)
            #print(delta)
            i += 1
        if self.actual_pos[-1][1] == "R":
            delta.append([time_to_meet(self.actual_pos[-1][0],self.end,extremity=True), [len(self.actual_pos)-1+shift]])

        self.list_events = delta
        if deb:
            print(start)
            print(self.actual_pos)
            print(self.list_events)
            print("End")
        return self.list_events
        #delta.sort()

    def rc(self,pos):
        #Relative coordinate
        return int(pos-self.start)

    def approx(self,p, r=7):
        return round(p, r)

    def evolvep(self,p, time, proba,d=1):

        #Evolve one fork at position p
        # with array of speed ,
        # for a given time


        def checkbound(p):
            return max(0,min(self.rc(p),len(self.mrt)-1))

        def assign_mrt_rfd(p,ev):
            pos = checkbound(p)
            if d == 1:
                t = self.t+ev
            else:
                t = self.t + ev #+ 1
            #print(pos, t)


            if np.isnan(self.mrt[pos]):

                self.mrt[pos] = t
                self.rfd[pos] = d
                proba[min(pos+self.start,len(proba)-1)]=0
            else:
                if self.mrt[pos] > t:
                    self.mrt[pos] = t
                    self.rfd[pos] = d
            proba[min(pos + self.start, len(proba) - 1)] = 0
            #else:

        # if forward

        r0 = p - int(p)

        if d == 1:
            ev = self.timespend[checkbound(int(p))] * (1 - r0)
        else:
            ev = self.timespend[checkbound(int(p))] * r0

        if ev <= time:
            p = int(p) + d
        else:
            p = p + d / self.timespend[checkbound(int(p))] * time
            assign_mrt_rfd(p,ev)
            return self.approx(p)

        while ev < time:
            assign_mrt_rfd(p,ev)

            ev += self.timespend[checkbound(int(p))]
            p += d


        # print(ev,time)
        if abs(ev - time) < 0.001:
            # print("equal")
            if d == -1:
                assign_mrt_rfd(p, ev)

                return p + 1
            else:

                assign_mrt_rfd(p, ev)
                return p
        else:
            if d == 1:
                p -= 1
                toev = (self.timespend[checkbound(int(p))] - (ev - time)) / self.timespend[checkbound(int(p))]
                return self.approx(p + d * toev)

            else:
                p += 1
                toev = (self.timespend[checkbound(int(p))] - (ev - time)) / self.timespend[checkbound(int(p))]
                return self.approx(p - d + d * toev)

    def evolve(self,time,proba,filter_termination=None):
        # evolve the chromosome of 'time' step
        remove = []
        termination = []
        avail = 0
        deb=False
        if deb:
            print("evolve",time)

        for d in self.list_events:
            deltat,particules = d
            d[0] -= time
            if round(d[0],3) <= 0:
                # if collision, avail + recompute list_events
                #print("Col",p)
                if len(particules) == 2:
                    avail += 1

                    delta_original = abs(self.actual_pos[particules[0]][2] - self.actual_pos[particules[1]][2])
                    if (filter_termination is None) or filter_termination <= delta_original:
                        termination.append(int(self.actual_pos[particules[0]][0]/2+self.actual_pos[particules[1]][0]/2))
                    if deb:
                        print(particules,termination)


                for part in particules:
                    remove.append(part)

        for p in self.actual_pos:
            #print(p[0], time)

            if p[1] == "L":

                if self.constant_speed:
                    self.rfd[self.rc(p[0] - time):self.rc(p[0])] = -1
                    self.mrt[self.rc(p[0] - time):self.rc(p[0])] = np.arange(self.t, self.t+time)[::-1]

                    proba[p[0] - time:p[0]]=0
                    p[0] -= time

                else:
                    previous = 0 + p[0]
                    p[0] = self.evolvep(p[0],time,proba,d=-1)


            else:
                if self.constant_speed:
                    self.rfd[self.rc(p[0]):self.rc(p[0] + time)] = 1
                    self.mrt[self.rc(p[0]):self.rc(p[0] + time)] = np.arange(self.t, self.t + time)

                    proba[p[0]:p[0] + time] = 0
                    p[0] += time


                else:
                    p[0] = self.evolvep(p[0], time, proba,d=1)

        delete = []
        if remove != []:
            #print("collision")
            #print(self.actual_pos)
            #print(self.list_events)
            #print(remove,len(self.actual_pos))
            for p in remove[::-1]:
                #print(p)
                delete.append(self.actual_pos.pop(p))
            self.list_events_comp()

        self.lt.extend(termination)
        self.t  += time

        # CHekc termination:
        for ter in termination:
            if np.isnan(self.mrt[self.rc(ter)]):
                self.check(show=True, msg="dt %f Termination not completed %.2f\n remve %s" % (time,ter,str(delete)))


        return termination,avail

    def min_time(self):
        #print("Le",self.list_events)
        mini = 1e6
        for d in self.list_events:
            if d[0] < mini:
                mini = round(d[0],7)

        #print(mini)
        return mini




    def check(self,show=False,msg=""):

        for p1, p2 in zip(self.actual_pos[:-1], self.actual_pos[1:]):
            if not(p1[0] <= p2[0]) or (not ([p1[1],p2[1]] in [["L","R"],["R","L"]])) or show:

                for actp in self.oldp:
                    print(actp)
                #print(self.oldp)
                print(self.start,self.end)
                print(p1, p2)

                print("List activation",self.la)
                print("List term",self.lt)
                print(self.list_events)
                print(self.actual_pos)
                import pylab
                f = pylab.figure()
                f.add_subplot(211)
                pylab.suptitle("RFD")
                pylab.plot(np.arange(self.start,self.end),self.rfd)
                f.add_subplot(212)
                pylab.plot(np.arange(self.start, self.end), self.mrt)
                print(msg)
                pylab.show()
                raise

        self.oldp.append(copy.deepcopy(self.actual_pos)+[])



def fast_rep(distrib, diff, debug=False, kon=0.001, fork_speed=0.3,
             single_mol_exp=False, single_mol_exp_sub_sample=50,
             pulse_size=2,cascade={},breaks=None,continuous=False,
             binsize=5,dori=30,list_chrom_ret=False,timespend=[],
             filter_termination=None,introduction_time=None,
             correct_activation=True,dario=False):

    #np.random.seed(0)
    pos = np.arange(len(distrib))

    if not continuous:
        init_ori = np.random.choice(pos, p=distrib/np.sum(distrib), size=int(len(distrib)*binsize/dori))
        init_distrib = np.zeros_like(distrib)
        # print(l)
        for p in init_ori:
            init_distrib[p] += 1
        #print(sum(init_distrib))
    else:
        init_distrib = distrib / np.sum(distrib)

    if dario:
        init_distrib = distrib
        #   print(sum(init_distrib))


    d3p= init_distrib

    introduced = 0

    if introduction_time is None:
        time = 0
        avail = diff
    else:
        time = 0 # very important because if not the first event is when the first diffusing element cover one ch

        avail = 1

    if correct_activation:
        avail=1

    proba = d3p.copy()

    pos = np.arange(len(d3p))

    single_mol_exp_v = []
    #print("SME", single_mol_exp)
    actual_pos = []
    list_chrom = []
    for start,end in breaks:
        if len(timespend) == 0:
            stimespend = []
        else:
            stimespend = timespend[start:end]
        list_chrom.append(Chrom(start,end,timespend=stimespend))

    def MRT():
        if len(list_chrom) == 1:
            return list_chrom[0].mrt
        else:
            return np.concatenate([c.mrt for c in list_chrom])

    def RFD():
        if len(list_chrom) == 1:
            return list_chrom[0].rfd
        else:
            return np.concatenate([c.rfd for c in list_chrom])

    #pos, proba, newp, finished = generate_newp(pos, proba, 1,actual_pos=[],cascade=cascade)
    #print("Before",avail)
    pos, proba, newp, finished,previous = generate_newp_no_corre(pos, proba, avail,
                                                                 actual_pos=actual_pos,cascade=cascade,
                                                                 previous =[])
    tot_introduced=avail
    nfork = [[avail,0]]

    avail = 0
    position_time_activated_ori = []
    terminations = []

    def in_ch_i(rpos):
        for i,(start,end) in enumerate(breaks):
            #print(start,end,pos)
            if start<=rpos< end:
                return i

    for p in newp:
        list_chrom[in_ch_i(p)].append_forks(p)
        position_time_activated_ori.append([p, 0, len(d3p)])

    finished = False
    if debug:
        print(actual_pos)

    count = 0
    smess = np.random.randint(0, single_mol_exp_sub_sample)
    smess_size = 0
    start_exp = False


    def next_event_breaks():

        return min( [chrom.min_time() for chrom in list_chrom])


    last_intro = 0
    continuous_time = 0
    Avails=[]
    Noris = []

    introduced=False
    #print("There")
    while (sum( [len(chrom.actual_pos) for chrom in list_chrom]) != 0) or np.any(proba!=0) :
        if avail > diff:
            print(avail,diff,time)

            raise
        if correct_activation and not introduced:
            avail = diff-1
            introduced=True
        #print(avail,time)

        Noris.append([np.sum(proba!=0),time])
        #print(time,avail,np.sum(proba!=0))


        #print(introduction_time,"INTSNST")
        count += 1
        smess += 1
        next_e = next_event_breaks()


        # evolve until next event :
        # either a fork collision
        # or an attachement
        if debug:
            print("Actual pos", actual_pos)
            print("next", next_e)

        find_target = False

        #print("Tmio",time,avail)



        #Gillespi algorithm
        fast=True
        if fast:

            def add_attach_nothing(attached=0,add=0,tot_introduced=0):

                nori = np.sum(proba != 0)

                if nori != 0 and avail - attached + add != 0:

                    k_attach = nori * (avail-attached+add) * kon  # minutes
                else:
                    # print("1001",nori,avail)
                    k_attach = 0


                if (introduction_time != None) and (tot_introduced < diff):
                    kaddN = 1.*diff / introduction_time * np.exp(-time / fork_speed(time) / introduction_time)
                else:
                    kaddN = 0

                #print("Inside",k_attach,kaddN)
                if k_attach !=0 or kaddN != 0:
                    next_attach_or_insert = - np.log(np.random.rand()) / (kaddN + k_attach)

                    if np.random.rand() < kaddN / (kaddN + k_attach):
                        return 0,1,next_attach_or_insert
                    else:
                        return 1,0,next_attach_or_insert

                else:
                    return 0,0,1000000000000000000

            attached = 0
            add = 0
            next_collide_time = next_e / fork_speed(time)
            passed_time = 0

            while True:
                status = add_attach_nothing(attached=attached,add=add,
                                            tot_introduced=tot_introduced)
                #print(status)
                if continuous_time + passed_time + status[2] * fork_speed(time) > time+ next_e:
                    time_evolve = next_e
                    continuous_time = time + time_evolve
                    #print("Event")
                    break
                attached += status[0]
                add += status[1]
                passed_time += status[2] * fork_speed(time)
                tot_introduced += status[1]

                if continuous_time + passed_time > time + 1:
                    #print("Plusone" )
                    time_evolve = int(continuous_time - time)
                    continuous_time += passed_time
                    break

                if tot_introduced == diff:
                    introduction_time = None

            #print("Avail %i, add %i , attached %i, tot_introduced %i"%(avail,add,attached,tot_introduced),continuous_time,time,time_evolve)
            if (introduction_time != None) and (time > (introduction_time * 3)) and tot_introduced != diff:
                avail += diff-tot_introduced
                tot_introduced = diff

            avail += add  # the one attached are remove afterward
            Avails.append([avail,time])
            nfork.append([nfork[-1][0],time])





        else:

            nori = np.sum(proba != 0)

            if nori != 0 and avail != 0:

                next_attach_time = 1 / (nori * avail * kon)  # minutes
            else:
                # print("1001",nori,avail)
                next_attach_time = 10000000

            kattach = 1/next_attach_time

            if introduction_time != None and tot_introduced <= diff:
                kaddN = diff/introduction_time * np.exp(-time / fork_speed(time)/introduction_time) #Ok for time unit
            else:
                kaddN = 0


            next_attach_or_insert = - np.log(np.random.rand()) / ( kaddN + kattach )

            release = False
            attached=0
            next_collide_time = next_e / fork_speed(time)  # minutes
            if next_attach_or_insert > next_collide_time:
                time_evolve = next_e
                continuous_time = time + time_evolve
                #print("Collision",time_evolve)

            else:

                if np.random.rand() < kaddN / (kaddN + kattach):
                    avail += 1
                    tot_introduced += 1
                else:
                    attached = 1

                continuous_time += (next_attach_or_insert * fork_speed(time))

                if continuous_time > time+1:
                    time_evolve = int(continuous_time-time)
                else:
                    time_evolve = 0
                #print("Reac",time_evolve,next_e)
                # print(next_attach_time)


        #print("time %.2f , %i avail %i nori, %.3f conti time , %i n intro , %i find target %i release "%(time,avail,np.sum(proba!=0),continuous_time,tot_introduced,find_target,release))

        """

            if nori == 0 or next_attach_time > next_collide_time:
                time_evolve = next_e  # kb
            else:
                time_evolve = int(next_attach_time * fork_speed(time)) # kb
                find_target = True
        """

        # print(collide,find_target)
        # print(next_attach_time,next_collide_time,avail,nori)
        # print(next_e[0][0])
        if time_evolve > next_e:
            print(time_evolve,next_e,continuous_time,time)
            raise

        # print(collide,find_target,avail)
        # print(actual_pos)
        # propagate

        #print(avail,update,add,continuous_time)


        if (time_evolve > 0 or next_collide_time == 0) : # Second condition not sure why but needed
        ##############################################################
        # To record single mol
            if single_mol_exp and (smess % single_mol_exp_sub_sample == 0):
                if not start_exp:
                    #print(time, "start")
                    single_mol_exp_v.append([time+time_evolve/2, time_evolve, np.sum(RFD() != 0)/len(d3p),
                                             copy.deepcopy(RFD().copy())])
                    start_exp = True
                else:
                    smess_size += time_evolve

            # To record single mol
            if single_mol_exp and (smess % single_mol_exp_sub_sample == 0):
                if smess_size > pulse_size:
                    single_mol_exp_v[-1][1] = smess_size
                    single_mol_exp_v[-1][-1] -= RFD().copy()
                    start_exp = False
                    smess_size = 0
                    #print(time, "end")

                else:
                    # print(ramp)
                    smess -= 1
            ##############################################################
            time += time_evolve
            actual_pos = []
            olds = np.sum(proba != 0)
            newavail =0
            for chrom in list_chrom:
                termination, iavail = chrom.evolve(time_evolve,proba,filter_termination=filter_termination)

                chrom.check()
                avail += iavail
                newavail += iavail

                actual_pos.extend(chrom.actual_pos)
                terminations.extend(termination)
            if newavail != 0:
                nfork.append([nfork[-1][0] - newavail, time])
                Avails.append([avail, time])

        if attached != 0:
            newp = []
            if not finished and np.sum(proba) != 0:
                #print("Reac",attached)
                # print("multiple",avail,toadd)
                # print(add)
                # pos, proba, newp, finished = generate_newp(pos, proba, 1,actual_pos=[],cascade=cascade)
                pos, proba, newp, finished, previous = generate_newp_no_corre(pos, proba, attached,
                                                                              actual_pos=actual_pos, cascade=cascade,
                                                                              previous=previous)

                # avail -= 1
            else:
                newp = []

            # Add the new one:
            MRTl = MRT()
            for p in newp:
                list_chrom[in_ch_i(p)].append_forks(p)
                proba[p] = 0
                position_time_activated_ori.append([p, time, len(d3p) - np.sum(~np.isnan(MRTl))])
                avail -= 1
            nfork.append([nfork[-1][0] + len(newp), time])
            Avails.append([avail,time])

        if tot_introduced ==60 and (nfork[-1][0]+Avails[-1][0]) < tot_introduced:
            print(nfork[-1][0],Avails[-1][0] , tot_introduced,len(breaks))
            print(newp)
            raise
        #print(avail,len(list_chrom[0].actual_pos)/2)
        if debug:
            print("AFter", actual_pos)

        for p1, p2 in zip(actual_pos[:-1], actual_pos[1:]):
            try:
                assert(p1[0] <= p2[0])
            except:
                print(p1, p2)
                raise

    if list_chrom_ret :

        return MRT(), RFD(), time, single_mol_exp_v, position_time_activated_ori,terminations , list_chrom
    else:
        return MRT(), RFD(), time, single_mol_exp_v, \
               position_time_activated_ori,terminations , \
               tot_introduced, Avails, Noris, nfork



def get_fast_MRT_RFDs(nsim, distrib, ndiff, dori=20, kon=0.001,
                      fork_speed=0.3,
                      single_mol_exp=False, pulse_size=5, it=True,
                      binsize=5,continuous=False,wholeRFD=False,cascade={},breaks=None,
                      n_jobs=6,timespend=[],nMRT=6,filter_termination=None,
                      introduction_time=None,
                      wholeMRT=False,return_dict=False,correct_activation=False,dario=False,mask=[],early_over_late=False):


    if not cascade:
        cascade = {}
    print("EXperimental")
    #np.random.seed(1)
    if breaks is None:
        breaks = [[0,len(distrib)]]
    MRTs = []
    RFDs = []
    Rep_Time = []
    single_mol_exp_vs = []
    position_time_activated_oris = []
    terminations = []
    tot_introduced = []
    Avails=[]
    Forks=[]

    Noris =[]
    #print("Nori", int(len(distrib)*binsize/dori))
    lao = []
    fork_speed_ct = False
    if type(fork_speed) in [float,int]:
        fs = 0 + fork_speed
        fork_speed = lambda x:fs
        fork_speed_ct = True

    from joblib import Parallel, delayed
    if n_jobs != 1:
        res = Parallel(n_jobs=n_jobs)(delayed(fast_rep)(distrib, ndiff, kon=kon, debug=False,
                            fork_speed=fork_speed, single_mol_exp=single_mol_exp,
                            pulse_size=pulse_size,cascade=cascade,breaks=breaks,
                            continuous=continuous,binsize=binsize,dori=dori,timespend=timespend,
                            filter_termination=filter_termination,
                            introduction_time=introduction_time,
                            correct_activation=correct_activation,dario=dario) for _ in range(nsim))
    else:
        res = [ fast_rep(distrib, ndiff, kon=kon, debug=False,
                            fork_speed=fork_speed, single_mol_exp=single_mol_exp,
                            pulse_size=pulse_size,cascade=cascade,breaks=breaks,
                            continuous=continuous,binsize=binsize,dori=dori,timespend=timespend,
                            filter_termination=filter_termination,introduction_time=introduction_time,
                            correct_activation=correct_activation,dario=dario) for _ in range(nsim)]

    for MRT, RFD, time, single_mol_exp_v, position_time_activated_ori,termination,totn,sAvails,sNoris,sForks in res:
        MRTs.append(MRT)
        RFDs.append(RFD)
        #Rep_Time.append(time)
        single_mol_exp_vs.append(single_mol_exp_v)
        # Rescale time in single mol to 0 1
        for i in range(len(single_mol_exp_vs[-1])):
            single_mol_exp_vs[-1][i][0] /= time
        position_time_activated_oris.append(position_time_activated_ori)
        terminations.append(termination)
        tot_introduced.append(totn)
        Avails.append(sAvails)
        Forks.append(sForks)
        Noris.append(sNoris)
        #print(sAvails[-4:],sForks[-4:],tot_introduced[-1])


    if it:
        for position_time_activated_ori in position_time_activated_oris:
            for p, t, unrep in position_time_activated_ori:
                lao.append([t, unrep])

    lao.sort()
    #print(len(lao))
    #print(lao)
    if fork_speed_ct:

        dt = 1 / fork_speed(0)  # in minute
        print("Fs cte , dt %.1f (min)"%dt)
    else:
        print("Dt 1")
        dt = 1

    maxi = int(round(lao[-1][0]))+1
    print("Average introduced" , np.mean(tot_introduced))
    print("Maxiiiiiiiiiiiiiii",maxi)

    Itun = np.zeros(maxi)
    Unrep = np.zeros(maxi) + np.nan
    It = np.zeros(maxi) + np.nan
    npts = np.zeros_like(It)

    for position_time_activated_ori in position_time_activated_oris:

        #print(position_time_activated_ori)

        for p, t, unrep in position_time_activated_ori:
            #Itun[int(t*dt)] += 1
            #print(p,t,int(t*dt),t*dt,unrep)
            assign = int(t)

            #left = t*dt-assign
            #if np.random.rand() < left:
            #    assign += 1
            #Unrep0[int(t*dt)] = unrep
            if np.isnan(It[assign]):
                It[assign] = 1 / unrep

            else:
                It[assign] += 1 / unrep

            if np.isnan(Unrep[assign]):
                Unrep[assign] = unrep

            else:
                Unrep[assign] += unrep

            npts[assign] += 1



    def compute_ft(data):
        #Not perfect are average is not done over all simus
        Flat_avail = np.zeros(maxi)
        Flat_N = np.zeros(maxi)
        for savails in data:
            #print(savails)
            for avail,ti in savails:
                if int(ti)<len(Flat_avail):
                    #if np.isnan()
                    Flat_avail[int(ti)] += avail
                    Flat_N[int(ti)] += 1
            for tip in range(ti,len(Flat_avail)):
                Flat_avail[int(tip)] += avail
                Flat_N[int(tip)] += 1
            #if int(ti) + 1 < len(Flat_avail):
            #    print(int(ti)+1)
            #    Flat_avail[int(ti) + 1] += avail
            #    Flat_N[int(ti)+1] += 1
        Flat_avail /= Flat_N
        return Flat_avail
        #print(It[:10])
    Flat_avail = compute_ft(Avails)
    Flat_ori = compute_ft(Noris)
    Flat_fork = compute_ft(Forks)


    # Probability of activation
    Pa = np.zeros_like(MRTs[0])
    for position_time_activated_ori in position_time_activated_oris:
        for p, _, _ in position_time_activated_ori:
            Pa[p] += 1

    Pa /= len(position_time_activated_oris)

    # Probability of terminations
    Pt = np.zeros_like(MRTs[0])
    for termination in terminations:
        for p in termination:
            #print(p)
            Pt[p] += 1

    Pt /= len(terminations)




    It = It / nsim /  binsize / dt # units /kb/min

    if early_over_late:
        nMRT=2

    n = nMRT
    dp = np.array(np.arange(0, 1+1/(2*n), 1/n)*100, dtype=np.int)
    MRTp = np.zeros_like(MRTs[0])
    MRTpstd = np.zeros((len(MRTs[0]),nMRT))



    for MRT in MRTs:
        if np.any(np.isnan(MRT)):
            MRT[np.isnan(MRT)]=np.nanmax(MRT)
        percentils = np.percentile(MRT, dp)
        for ip, (p1, p2) in enumerate(zip(percentils[:-1], percentils[1:]), 1):
            MRTp[(MRT > p1) & (MRT <= p2)] += ip/n
            MRTpstd[(MRT > p1) & (MRT <= p2),ip-1] += 1
        MRTp[MRT == 0] += 1/n
        MRTpstd[MRT == 0,0] += 1

    #MRT_normed = MRTp / nsim * n / (n - 1) - 1 / (n - 1)
    MRT_normed = MRTp/nsim - 1/(2*n)
    MRTpstd = MRTpstd / nsim

    if early_over_late:
        MRT_normed = MRTpstd[::,0]/(MRTpstd[::,1])
        MRT_normed[MRTpstd[::,1]==0]=np.nan
        #MRT_normed /= np.max(MRT_normed)
    #print(np.sum(MRTpstd,axis=1))

    def delta_std(x):
        time = np.arange(nMRT) / (nMRT-1)
        time = time[np.newaxis, ::] * np.ones((len(x)))[::, np.newaxis]
        mrt = np.sum(x * time, axis=1)[::, np.newaxis]
        return np.sum(x * (time - mrt) ** 2, axis=1) ** 0.5
    MRTpstd = delta_std(MRTpstd)

    if len(mask) == 0:
        mask=np.ones_like(MRT_normed)

    def T(mrt, p, symetric=True):

        if p == 0:
            return np.nanmax(mrt*mask)
        if not symetric:
            return np.nanpercentile(mrt*mask, 100 - p) - np.nanmin(mrt*mask)
        else:
            return np.nanpercentile(mrt*mask, 100 - p / 2) - np.nanpercentile(mrt*mask, p / 2)

    #print(mask)
    Rep_Time = [T(mrt,p=0,symetric=False) * dt for mrt in MRTs]
    #print(Rep_time)
    T99 = [T(mrt,p=1,symetric=False) * dt for mrt in MRTs]
    T95 = [T(mrt,p=5,symetric=False) * dt for mrt in MRTs]

    if return_dict:
        return {"mean_MRT_normed":MRT_normed,
                "MRTpstd":MRTpstd,
                "mean_MRT_time":np.mean(np.array(MRTs), axis=0) * dt,
                "hist_MRT":[MRTi * dt for MRTi in MRTs],
                "mean_RFD":np.mean(np.array(RFDs), axis=0),
                "hist_RFD":RFDs,
                "hist_replication_time":np.array(Rep_Time),
                 "single_mol_exp":single_mol_exp_vs,
                "position_time_activated_oris":position_time_activated_oris,
                "It":It,"proba_activation":Pa,"proba_termination":Pt,
                 "time":np.arange(len(It))*dt,
                 "Free":Flat_avail,
                "Nori": Flat_ori
                }

    if not wholeRFD:
        return MRT_normed, np.mean(np.array(MRTs), axis=0), \
            np.mean(np.array(RFDs), axis=0), np.array(Rep_Time) , single_mol_exp_vs, \
            position_time_activated_oris, It , Pa, Pt, np.arange(len(It))*dt,MRTpstd , \
               Flat_avail,Flat_fork,[MRTi * dt for MRTi in MRTs],T99,T95

    if not wholeMRT:
        return MRT_normed, np.mean(np.array(MRTs), axis=0), \
            np.mean(np.array(RFDs), axis=0), np.array(Rep_Time) , single_mol_exp_vs, \
            position_time_activated_oris, It,np.array(RFDs)

    return MRT_normed, np.mean(np.array(MRTs), axis=0), \
               np.mean(np.array(RFDs), axis=0), np.array(Rep_Time), single_mol_exp_vs, \
               position_time_activated_oris, It, np.array(RFDs),np.array(MRTs)
