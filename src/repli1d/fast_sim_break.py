import numpy as np
import copy
#np.random.seed(0)
nc = 0


def generate_newp_cascade(pos, proba, avail,actual_pos=[]):
    #print("cascade",actual_pos)
    newp = []
    finished = False

    def comupute_enhanced(proba,actual_pos,infl=10,amount=4,down=0,damount=0):
        cproba = np.zeros_like(proba) + 1
        for position, direction in actual_pos:
            #print(position,direction,cproba,max(position-infl),position)
            if direction == "L":
                cproba[position:min(position+infl,len(cproba)-1)] += amount
            if direction == "R":
                cproba[max(position - infl,0):position] += amount

            if direction == "L":
                cproba[position:min(position+down,len(cproba)-1)] =0
            if direction == "R":
                cproba[max(position - down,0):position] += 0

        return cproba

    cproba = comupute_enhanced(proba, actual_pos, infl=30, amount=200)
    filter = proba != 0

    npossible = np.sum(filter)

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


def generate_newp_no_corre(pos, proba, avail,actual_pos=[],cascade=False,previous =[]):

    #Return a list of #avail site
    #Modify proba and previous

    if cascade:
        pos, proba, newp, finished = generate_newp_cascade(pos, proba, avail, actual_pos=actual_pos)
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
    def __init__(self,start,end):
        self.start = start
        self.end = end
        self.actual_pos = []
        self.list_events = []
        self.delta = []
        self.rfd = np.zeros(end-start)
        self.debug = False
        self.lt = []
        self.list_pause = []

    def append_forks(self,pos):
        if self.debug:
            print("New Fork",pos)
        found=False
        self.list_evens = []
        #print(self.actual_pos)
        if len(self.actual_pos) != 0 and self.actual_pos[0][0] > pos:
            if self.list_events != []:
                self.list_events[0] = [self.actual_pos[0][0]-pos,[1,2]]
                self.list_events.insert(0,[pos,[0]])
                #for p in self.list_events[2:]:


            self.actual_pos.insert(0, [pos, "L"])
            self.actual_pos.insert(1, [pos, "R"])
            found = True

        for p2 in range(len(self.actual_pos)-1):
            if self.actual_pos[p2+1][0] > pos >= self.actual_pos[p2][0]:
                self.actual_pos.insert(p2+1, [pos, "L"])
                self.actual_pos.insert(p2+2, [pos, "R"])
                found = True
                break
        if not found:
            self.actual_pos.extend([[pos, "L"], [pos, "R"]])


        if self.list_events == []:
            self.list_events.append([pos-self.start,[0]])
            self.list_events.append([self.end-pos,[1]])
        #print(self.actual_pos)
        #self.check()
        self.list_events_comp()


    def list_events_comp(self,shift=0):
        if self.actual_pos == []:
            self.list_events = []
            return []
        delta = []
        start = 0
        end = None
        if self.actual_pos[0][1] == "L":
            delta.append([self.actual_pos[0][0]-self.start, [0+shift]])
            if self.actual_pos[0][0]-self.start < 0:
                raise
            start = 1
        if self.actual_pos[-1][1] == "R":
            end = -1
        i = 0
        # if not (len(delta) == 2 and len(actual_pos) == 2):
        for p1, p2 in zip(self.actual_pos[start:end][::2], self.actual_pos[start:end][1::2]):
            deltapt = p2[0]-p1[0]
            if deltapt % 2 == 0:
                deltapt = (p2[0]-p1[0])//2
            else:
                deltapt = (p2[0] - p1[0]) // 2 + 1
            delta.append([deltapt, [2*i+start+shift, 2*i+1+start+shift]])
            #print(p2[0] ,p1[0],(p2[0]-p1[0])//2+1)
            #print(delta)
            #print(delta)
            i += 1
        if self.actual_pos[-1][1] == "R":
            delta.append([self.end-self.actual_pos[-1][0], [len(self.actual_pos)-1+shift]])

        self.list_events = delta
        return self.list_events
        #delta.sort()

    def evolve(self,time):
        #print("evol",time)
        remove = []
        termination = []
        avail = 0
        for d in self.list_events:
            deltat,particules = d
            d[0] -= time
            if d[0] <= 0:
                # if collision, avail + recompute list_events
                #print("Col",p)
                if len(particules) == 2:
                    avail += 1
                    #print(particules)
                    termination.append(int(self.actual_pos[particules[0]][0]/2+self.actual_pos[particules[1]][0]/2))

                for part in particules:
                    remove.append(part)

        for p in self.actual_pos:

            if p[1] == "L":
                self.rfd[p[0]-time:p[0]] = -1
                p[0] -= time

            else:
                self.rfd[p[0]:p[0] + time] = 1
                p[0] += time

        if remove != []:
            #print("collision")
            #print(self.actual_pos)
            #print(self.list_events)
            #print(remove,len(self.actual_pos))
            for p in remove[::-1]:
                #print(p)
                self.actual_pos.pop(p)
            self.list_events_comp()

        self.lt.extend(termination)

        return termination,avail

    def min_time(self):
        #print("Le",self.list_events)
        mini = 1e6
        for d in self.list_events:
            if d[0] < mini:
                mini = d[0]
        return mini

    def check(self):


        for p1, p2 in zip(self.actual_pos[:-1], self.actual_pos[1:]):
            try:
                assert(p1[0] <= p2[0])
                assert([p1[1],p2[1]] in [["L","R"],["R","L"]])
            except:
                print(self.start,self.end)
                print(p1, p2)
                print(self.lt)
                print(self.oldp)
                print(self.actual_pos)
                raise
        self.oldp = self.actual_pos + []



def fast_rep(distrib, diff, debug=False, kon=0.001, fork_speed=0.3,
             single_mol_exp=False, single_mol_exp_sub_sample=50,
             pulse_size=2,cascade=False,breaks=None,continuous=False,binsize=5,dori=30,list_chrom_ret=False):


    pos = np.arange(len(distrib))

    if not continuous:
        init_ori = np.random.choice(pos, p=distrib/np.sum(distrib), size=int(len(distrib)*binsize/dori))
        init_distrib = np.zeros_like(distrib)
        # print(l)
        for p in init_ori:
            init_distrib[p] += 1
    else:
        init_distrib = distrib / np.sum(distrib)

    d3p= init_distrib

    MRT = np.zeros_like(d3p)
    RFD = np.zeros_like(d3p)


    avail = diff
    proba = d3p.copy()

    pos = np.arange(len(d3p))

    single_mol_exp_v = []
    #print("SME", single_mol_exp)
    actual_pos = []
    list_chrom = []
    for start,end in breaks:
        list_chrom.append(Chrom(start,end))

    #pos, proba, newp, finished = generate_newp(pos, proba, 1,actual_pos=[],cascade=cascade)
    pos, proba, newp, finished,previous = generate_newp_no_corre(pos, proba, avail,
                                                                 actual_pos=actual_pos,cascade=cascade,
                                                                 previous =[])
    avail = 0
    position_time_activated_ori = []
    terminations = []

    time = 0
    def in_ch_i(rpos):
        for i,(start,end) in enumerate(breaks):
            #print(start,end,pos)
            if start<=rpos< end:
                return i

    for p in newp:
        list_chrom[in_ch_i(p)].append_forks(p)
        MRT[p] = time
        position_time_activated_ori.append([p, 0, len(MRT)])

    finished = False
    if debug:
        print(actual_pos)

    count = 0
    smess = np.random.randint(0, single_mol_exp_sub_sample)
    smess_size = 0
    start_exp = False


    def next_event_breaks():

        return min( [chrom.min_time() for chrom in list_chrom])



    while sum( [len(chrom.actual_pos) for chrom in list_chrom]) != 0:
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
        if avail == 0:
            time_evolve = next_e # Fork speed ?
        else:
            nori = np.sum(proba != 0)

            if nori != 0:
                next_attach_time = 1/((np.sum(proba != 0) * avail * kon)) # minutes
            next_collide_time = next_e / fork_speed(time)   # minutes

            if nori == 0 or next_attach_time > next_collide_time:
                time_evolve = next_e  # kb
            else:
                time_evolve = int(next_attach_time * fork_speed(time)) # kb
                find_target = True


            # print(collide,find_target)
            # print(next_attach_time,next_collide_time,avail,nori)
            # print(next_e[0][0])
            assert(time_evolve <= next_e)

        # print(collide,find_target,avail)
        # print(actual_pos)
        # propagate
        ramp = np.arange(time, time+time_evolve)

        ##############################################################
        # To record single mol
        if single_mol_exp and (smess % single_mol_exp_sub_sample == 0):
            if not start_exp:
                #print(time, "start")
                single_mol_exp_v.append([time+time_evolve/2, time_evolve, np.sum(MRT != 0)/len(MRT),
                                         copy.deepcopy(RFD.copy())])
                start_exp = True
            else:
                smess_size += time_evolve
        ##############################################################

        #print(next_e,time_evolve,find_target)
        for chrom in list_chrom:
            #print(chrom.actual_pos)
            #chrom.check()
            for p in chrom.actual_pos:
                if p[1] == "L":
                    # print(p[0]-time_evolve,p[0])
                    # print(proba[0])
                    MRT[p[0]-time_evolve:p[0]] = ramp[::-1]
                    proba[p[0]-time_evolve:p[0]] = 0
                    RFD[p[0]-time_evolve:p[0]] = -1
                    proba[p[0]] = 0
                else:
                    #print(time_evolve,p,len(ramp))
                    MRT[p[0]:p[0]+time_evolve] = ramp
                    proba[p[0]:p[0]+time_evolve] = 0
                    RFD[p[0]:p[0]+time_evolve] = +1
                    if p[0] < len(proba):
                        proba[p[0]] = 0

        ##############################################################
        # To record single mol
        if single_mol_exp and (smess % single_mol_exp_sub_sample == 0):
            if smess_size > pulse_size:
                single_mol_exp_v[-1][1] = smess_size
                single_mol_exp_v[-1][-1] -= RFD.copy()
                start_exp = False
                smess_size = 0
                #print(time, "end")

            else:
                # print(ramp)
                smess -= 1
        ##############################################################
        time += time_evolve
        actual_pos = []
        for chrom in list_chrom:
            termination, iavail = chrom.evolve(time_evolve)
            chrom.check()
            avail += iavail
            actual_pos.extend(chrom.actual_pos)
            terminations.extend(termination)
        # print()
        if find_target:

            if not finished and np.sum(proba) != 0:

                #pos, proba, newp, finished = generate_newp(pos, proba, 1,actual_pos=[],cascade=cascade)
                pos, proba, newp, finished,previous = generate_newp_no_corre(pos, proba, avail,
                                                                             actual_pos=actual_pos,cascade=cascade,
                                                                             previous =previous)
            else:
                newp = []

            # Add the new one:
            for p in newp:
                list_chrom[in_ch_i(p)].append_forks(p)
                MRT[p] = time
                position_time_activated_ori.append([p, time,  len(MRT) - np.sum(MRT != 0)])
                avail -= 1
        if debug:
            print("AFter", actual_pos)

        for p1, p2 in zip(actual_pos[:-1], actual_pos[1:]):
            try:
                assert(p1[0] <= p2[0])
            except:
                print(p1, p2)
                raise



        #print("Here",avail,np.sum([len(chrom.actual_pos)//2 for chrom in list_chrom]))
        #print(avail,time)
        # break

        # plot(MRT,label=str(count))
        # legend()
        #if count == 2000:
        #    print("Ended")
        #    break
    #print("Nrun",count)
    if list_chrom_ret :

        return MRT, RFD, time, single_mol_exp_v, position_time_activated_ori,terminations , list_chrom
    else:
        return MRT, RFD, time, single_mol_exp_v, position_time_activated_ori,terminations



def get_fast_MRT_RFDs(nsim, distrib, ndiff, dori=20, kon=0.001,
                      fork_speed=0.3,
                      single_mol_exp=False, pulse_size=5, it=True,
                      binsize=5,continuous=False,wholeRFD=False,cascade=False,breaks=None,n_jobs=6):

    print("EXperimental")
    if breaks is None:
        breaks = [[0,len(distrib)]]
    MRTs = []
    RFDs = []
    Rep_Time = []
    single_mol_exp_vs = []
    position_time_activated_oris = []
    terminations = []
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
                            continuous=continuous,binsize=binsize,dori=dori) for _ in range(nsim))
    else:
        res = [ fast_rep(distrib, ndiff, kon=kon, debug=False,
                            fork_speed=fork_speed, single_mol_exp=single_mol_exp,
                            pulse_size=pulse_size,cascade=cascade,breaks=breaks,
                            continuous=continuous,binsize=binsize,dori=dori) for _ in range(nsim)]

    for MRT, RFD, time, single_mol_exp_v, position_time_activated_ori,termination in res:
        MRTs.append(MRT)
        RFDs.append(RFD)
        Rep_Time.append(time)
        single_mol_exp_vs.append(single_mol_exp_v)
        position_time_activated_oris.append(position_time_activated_ori)
        terminations.append(termination)


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

    maxi = int(lao[-1][0]*dt)+1
    print("Maxiiiiiiiiiiiiiii",maxi)
    Itun = np.zeros(maxi)
    Unrep = np.zeros(maxi)
    It = np.zeros(maxi) + np.nan


    for position_time_activated_ori in position_time_activated_oris:
        Unrep0 =  np.zeros(maxi)
        for p, t, unrep in position_time_activated_ori:
            Itun[int(t*dt)] += 1
            Unrep0[int(t*dt)] = unrep
            if np.isnan(It[int(t*dt)]):
                It[int(t*dt)] = 1/unrep
            else:
                It[int(t*dt)] += 1/unrep
        Unrep += Unrep0 / nsim


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

    n = 6
    dp = np.array(np.arange(0, 1+1/(2*n), 1/n)*100, dtype=np.int)
    MRTp = np.zeros_like(MRTs[0])
    for MRT in MRTs:
        percentils = np.percentile(MRT, dp)
        for ip, (p1, p2) in enumerate(zip(percentils[:-1], percentils[1:]), 1):
            MRTp[(MRT > p1) & (MRT <= p2)] += ip/n
        MRTp[MRT == 0] += 1/n
    if not wholeRFD:
        return MRTp/nsim * n / (n - 1) - 1 / (n - 1), np.mean(np.array(MRTs), axis=0), \
            np.mean(np.array(RFDs), axis=0), np.array(Rep_Time) * dt, single_mol_exp_vs, \
            position_time_activated_oris, It , Pa, Pt
    else:
        return MRTp/nsim * n / (n - 1) - 1 / (n - 1), np.mean(np.array(MRTs), axis=0), \
            np.mean(np.array(RFDs), axis=0), np.array(Rep_Time) * dt, single_mol_exp_vs, \
            position_time_activated_oris, It,np.array(RFDs)
