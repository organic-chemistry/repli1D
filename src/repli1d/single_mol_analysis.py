import numpy as np

def compute_real_inter_ori(pos_t_activated_ori):
#print(pos)
    deltas = []
    #print(len(pos))
    #print(pos[0])
    for ipos in pos_t_activated_ori:
        ipos.sort()
        ipos=np.array(ipos)
        deltas.append(ipos[1:] - ipos[:-1])
        deltas[-1] = deltas[-1][::,0]
    return np.concatenate(deltas)

def compute_mean_activated_ori(pos_t_activated_ori):
    #pos_t_activated_ori is a list of length n simu.
    #each element is a list with:
    #position, time,  unreplicated DNA

    mean_activated = []
    #print(len(pos))
    #print(pos[0])
    for ipos in pos_t_activated_ori:
        #ipos.sort()
        #ipos=np.array(ipos)
        #deltas.append(ipos[1:] - ipos[:-1])
        #deltas[-1] = deltas[-1][::,0]
        mean_activated.append(len(ipos))
    return np.mean(mean_activated)


def get_parameters_one_fiber(onef):
    """
    Onef is a delta RFD between two time points.
    for a fixed pulse length
    """

    #First sanitize
    #Remove +2 or -2 that correspond to merging of forks
    onef[onef == -2] = 0
    onef[onef == 2] = 0
    onef=np.concatenate([[0],onef,[0]])

    delta = onef[1:]-onef[:-1]

    events = [int(idelta) for idelta in delta if idelta != 0]

    if len(events) == 0:
        return {"dist_ori":[],"size_forks":[],"n_ori":0}
    positions = [pos for pos, idelta in enumerate(delta) if idelta != 0]

    #print(events)
    #print(positions)
    e_str=",".join([str(e) for e in events])

    #Remove evything aroundcollision
    e_str=e_str.replace("-1,2,1","0,2,0")
    e_str=e_str.replace("-1,2,-2","0,2,0")
    e_str=e_str.replace("-2,2,1","0,2,0")
    e_str=e_str.replace("-2,2,-2","0,2,0")

    e_str=e_str.replace("1,-2,1","0,-2,0")
    #print(e_str)


    e_str=e_str.replace("-1,1","R,0")
    e_str=e_str.replace("1,-1","0,L")


    events=e_str.split(",")
    #print(events)
    #print(len(events),len(positions))
    assert(len(events)==len(positions))

    size_forks = []

    for p,e in zip(positions,events):
        #print(e,p)
        if e == "L":
            ip = 0 + p
            size_fork = 0
            #print(onef[p])
            while ip >= 0 and onef[ip] == 1:
                size_fork += 1
                ip -= 1
            #print("SF",size_fork)
            size_forks.append(size_fork)
        if e == "R":
            ip = 0 + p +1
            size_fork = 0
            #print(onef[p])
            while ip < len(onef) and onef[ip] == -1:
                size_fork += 1
                ip += 1
            #print("SF",size_fork)
            size_forks.append(size_fork)
            #print(onef[p])

    n_ori = 0
    for i,(e1,e2) in enumerate(zip(events[:-1],events[1:])):
        if e1 == "L" and e2 == "R":
            #print(positions[i],positions[i+1],"p")
            #get length
            events[i]="O_%.1f"%(positions[i+1]/2+positions[i]/2)
            events[i+1]="0"


    #print("events",events)

    orip = []
    for i,e in enumerate(events):
        if e == "-2":
            # Fork that are not separated
            n_ori += 1
            orip.append(positions[i])
            size_forks.append(1)
            size_forks.append(1)

        elif "O" in e:
            orip.append(float(e.split("_")[-1]))
            n_ori += 1

        elif e == "2":
            # Termination
            size_forks.append(1)
            size_forks.append(1)
            if orip != [] and orip[-1] != "B":
                orip.append("B")  # collisions

        else:
            if e != "0":
                if orip != [] and orip[-1] != "B":
                    orip.append("B")  #collisions

    delta = []
    for p,p1 in zip(orip[:-1],orip[1:]):
        if p != "B" and p1 != "B":
            delta.append(p1-p-1)

    return {"dist_ori":delta,"size_forks":size_forks,"n_ori":n_ori}


def compute_info(Sme, fraction=[],size_single_mol=None,n_sample=1000):
    info = {"dist_ori":[],"n_ori":[],"size_forks":[]}
    for sme in Sme:
        for sgl in sme:
            if fraction != []:
                if sgl[2] < fraction[0] or sgl[2] > fraction[1]:
                    continue

            start_end = []

            if type(sgl[-1]) != list:
                #Only one contig
                if size_single_mol is None:
                    start_end = [[0,None]]
                else:
                    for s in np.random.randint(0,len(sgl[-1])-size_single_mol-1,n_sample):
                        start_end.append([s,s+size_single_mol])
                #print(start_end)
                for s,e in start_end:
                    deltas = get_parameters_one_fiber(sgl[-1][s:e].copy())
                    #print(deltas)
                    for k in info.keys():
                        info[k].append(deltas[k])
            else:
                #print(sgl)
                #Multiple contigs
                if size_single_mol is None:
                    start_end = []
                    for contig in range(len(sgl[-1])):
                        start_end += [[contig,0,None]]
                else:
                    #print(sgl)
                    for contig in range(len(sgl[-1])):
                        if len(sgl[-1][contig])-size_single_mol-1 > 0:
                            for s in np.random.randint(0,len(sgl[-1][contig])-size_single_mol-1,n_sample//len(sgl[-1])):
                                start_end.append([contig,s,s+size_single_mol])
                #print(start_end)
                for contig,s,e in start_end:
                    deltas = get_parameters_one_fiber(sgl[-1][contig][s:e].copy())
                    #print(deltas)
                    for k in info.keys():
                        info[k].append(deltas[k])

    return info
