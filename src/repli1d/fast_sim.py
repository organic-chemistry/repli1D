import numpy as np
import copy


def generate_newp(pos, proba, avail,actual_pos=[],cascade=False):
    newp = []
    finished = False

    def comupute_enhanced(proba,actual_pos,infl=10,amount=4,down=3,damount=0):
        cproba = np.zeros_like(proba) + 1
        for position, direction in actual_pos:
            #print(position,direction,cproba,max(position-infl),position)
            if direction == "L":
                cproba[position:min(position+infl,len(cproba)-1)] += amount
            if direction == "R":
                cproba[max(position -infl,0):position] += amount

            if direction == "L":
                cproba[position:min(position+down,len(cproba)-1)] =0
            if direction == "R":
                cproba[max(position -down,0):position] += 0

        return cproba

    cproba = np.zeros_like(proba) + 1

    for p in range(avail):
        if cascade:
            cproba = comupute_enhanced(proba,actual_pos,infl=30,amount=20)
            #print(np.mean(cproba))
        proba /= np.sum(proba)
        filter = proba!=0
        #print(np.sum(filter))
        newp.append(np.random.choice(pos[filter], p=proba[filter]*cproba[filter] / (np.sum(cproba[filter] * proba[filter]))))
        proba[newp[-1]] = 0
        if np.sum(proba) == 0:
            finished = True
            break

    newp.sort()

    return pos, proba, newp, finished


def next_event(actual_pos, total_size):
        # check extremities
    delta = []
    start = 0
    end = None
    if actual_pos[0][1] == "L":
        delta.append([actual_pos[0][0], [0]])
        start = 1
    if actual_pos[-1][1] == "R":
        delta.append([total_size-actual_pos[-1][0], [len(actual_pos)-1]])
        end = -1
    i = 0
    # if not (len(delta) == 2 and len(actual_pos) == 2):
    for p1, p2 in zip(actual_pos[start:end][::2], actual_pos[start:end][1::2]):
        delta.append([(p2[0]-p1[0])//2+1, [2*i+start, 2*i+1+start]])
        i += 1
    delta.sort()
    return delta


def fast_rep(d3p, diff, debug=False, kon=0.001, fork_speed=0.3,
             single_mol_exp=False, single_mol_exp_sub_sample=50, pulse_size=5,cascade=False):

    MRT = np.zeros_like(d3p)
    RFD = np.zeros_like(d3p)


    avail = diff
    proba = d3p.copy()

    pos = np.arange(len(d3p))

    single_mol_exp_v = []
    #print("SME", single_mol_exp)
    actual_pos = []

    pos, proba, newp, finished = generate_newp(pos, proba, 1,actual_pos=actual_pos,cascade=cascade)
    avail = max(0,avail-1)
    position_time_activated_ori = []

    time = 0
    for p in newp:
        actual_pos.append([p, "L"])
        actual_pos.append([p, "R"])
        MRT[p] = time

        position_time_activated_ori.append([p, 0, len(MRT)])

    finished = False
    if debug:
        print(actual_pos)

    count = 0
    smess = np.random.randint(0, single_mol_exp_sub_sample)
    smess_size = 0
    start_exp = False

    while actual_pos != []:
        count += 1
        smess += 1
        next_e = next_event(actual_pos, total_size=len(pos))

        # evolve until next event :
        # either a fork collision
        # or an attachement

        if debug:
            print("Actual pos", actual_pos)
            print("next", next_e)

        collide = False
        find_target = False
        if avail == 0:
            time_evolve = next_e[0][0] # Fork speed ?
            collide = True
        else:
            nori = np.sum(proba != 0)

            if nori != 0:
                next_attach_time = 1/((np.sum(proba != 0) * avail * kon))
            next_collide_time = next_e[0][0] / fork_speed(time)

            if nori == 0 or next_attach_time > next_collide_time:
                collide = True
                time_evolve = next_e[0][0]
            else:
                time_evolve = int(next_attach_time * fork_speed(time))
                find_target = True
                if time_evolve == next_e[0][0]:
                    collide = True

            # print(collide,find_target)
            # print(next_attach_time,next_collide_time,avail,nori)
            # print(next_e[0][0])
            assert(time_evolve <= next_e[0][0])

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
        for p in actual_pos:
            if p[1] == "L":
                # print(p[0]-time_evolve,p[0])
                # print(proba[0])
                MRT[p[0]-time_evolve:p[0]] = ramp[::-1]
                proba[p[0]-time_evolve:p[0]] = 0
                RFD[p[0]-time_evolve:p[0]] = -1
                p[0] -= time_evolve
                proba[p[0]] = 0
            else:
                MRT[p[0]:p[0]+time_evolve] = ramp
                proba[p[0]:p[0]+time_evolve] = 0
                RFD[p[0]:p[0]+time_evolve] = +1
                p[0] += time_evolve
                if p[0] < len(proba):
                    proba[p[0]] = 0
        time += time_evolve

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

        if collide:
            to_remove = []
            for delta_t, whats in next_e:
                if debug:
                    print(whats)
                if delta_t != time_evolve:
                    break
                to_remove.extend(whats)
                if len(whats) == 2:
                    avail += 1
            to_remove.sort()
            if debug:
                print("RM", to_remove)
            for p in to_remove[::-1]:
                actual_pos.pop(p)

        # print()
        if find_target:

            if not finished and np.sum(proba) != 0:
                pos, proba, newp, finished = generate_newp(pos, proba, 1,actual_pos=actual_pos,cascade=cascade)
            else:
                newp = []

            # Add the new one:
            if debug:
                print(len(newp), len(actual_pos))
                print(newp)
                print("Before", actual_pos)
            for p in newp:
                position_time_activated_ori.append([p, time, len(MRT) - np.sum(MRT != 0)])

                # print(p)
                found = False
                if len(actual_pos) != 0 and actual_pos[0][0] > p:
                    actual_pos.insert(0, [p, "L"])
                    actual_pos.insert(1, [p, "R"])
                    break
                    found = True
                for p2 in range(len(actual_pos)-1):
                    if actual_pos[p2+1][0] > p > actual_pos[p2][0]:
                        actual_pos.insert(p2+1, [p, "L"])
                        actual_pos.insert(p2+2, [p, "R"])
                        found = True
                if not found:
                    actual_pos.extend([[p, "L"], [p, "R"]])
                avail -= 1
        if debug:
            print("AFter", actual_pos)

        for p1, p2 in zip(actual_pos[:-1], actual_pos[1:]):
            try:
                assert(p1[0] <= p2[0])
            except:
                print(p1, p2)
                raise
        # break

        # plot(MRT,label=str(count))
        # legend()
        if count == 2000:
            print("Ended")
            break

    return MRT, RFD, time, single_mol_exp_v, position_time_activated_ori


def get_fast_MRT_RFDs(nsim, distrib, ndiff, dori=20, kon=0.001,
                      fork_speed=0.3,
                      single_mol_exp=False, pulse_size=5, it=True, binsize=5,continuous=False,wholeRFD=False,cascade=False):
    MRTs = []
    RFDs = []
    Rep_Time = []
    single_mol_exp_vs = []
    position_time_activated_oris = []
    #print("Nori", int(len(distrib)*binsize/dori))
    lao = []
    fork_speed_ct = False
    if type(fork_speed) in [float,int]:
        fs = 0 + fork_speed
        fork_speed = lambda x:fs
        fork_speed_ct = True
    for i in range(nsim):
        pos = np.arange(len(distrib))

        if not continuous:
            init_ori = np.random.choice(pos, p=distrib/np.sum(distrib), size=int(len(distrib)*binsize/dori))
            init_distrib = np.zeros_like(distrib)
            # print(l)
            for p in init_ori:
                init_distrib[p] += 1
        else:
            init_distrib = distrib / np.sum(distrib)
        # print(init_ori)



        MRT, RFD, time, single_mol_exp_v, position_time_activated_ori = fast_rep(init_distrib, ndiff, kon=kon, debug=False,
                                                                                 fork_speed=fork_speed, single_mol_exp=single_mol_exp,
                                                                                 pulse_size=pulse_size,cascade=cascade)
        MRTs.append(MRT)
        RFDs.append(RFD)
        Rep_Time.append(time)
        single_mol_exp_vs.append(single_mol_exp_v)
        position_time_activated_oris.append(position_time_activated_ori)

        if it:
            for p, t, unrep in position_time_activated_ori:
                lao.append([t, unrep])

    lao.sort()
    #print(len(lao))
    if fork_speed_ct:

        dt = 1/fork_speed(0)  # in minute
        print("Fs cte , dt %.1f (min)"%dt)
    else:
        print("Dt 1")
        dt = 1

    maxi = int(lao[-1][0]*dt)+1
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

    It = It / nsim /  binsize / dt # units /kb/min

    """
    for t, unrep in lao:
        # print(t)
        Itun[int(t*dt)] += 1
        Unrep[int(t*dt)] += unrep #Before was doing += ...
        It[int(t*dt)] += 1/unrep

    It = Itun / nsim / ((Unrep) * binsize) / dt # units /kb/min
    #It = It /  nsim /   binsize / dt # units /kb/min
    """
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
            position_time_activated_oris, It
    else:
        return MRTp/nsim * n / (n - 1) - 1 / (n - 1), np.mean(np.array(MRTs), axis=0), \
            np.mean(np.array(RFDs), axis=0), np.array(Rep_Time) * dt, single_mol_exp_vs, \
            position_time_activated_oris, It,np.array(RFDs)
