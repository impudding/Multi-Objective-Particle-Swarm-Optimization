#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:38:59 2019

@author: yun-chen
"""
import random
import numpy as np
import pandas as pd
import datetime
import copy
"""
 pseudo code for PSO
for particle i:
        for dimension d:
            initialize position Xid randomly in some range
            initialize velocity Vid randomly in some range

iteration k = 1

' find P_BEST '

for particle i:
    calculate fitness value
    if fitness is better than P_BESTid
        P_BEStid = fitness            

' find G_BEST '

for particle i:
    find the G_BEST
    
' move the particles '

for particle i:
    for dimension d:
        Vid(k + 1) = (w * Vid(k)) + (c1 * rand1 * (Pid - Xid)) + (c2 * rand2 * (Gid - Xid))
        Xid(k + 1) = Xid(k) + Vid(k + 1)
        
' next iteration
k = k + 1
"""
starttime = datetime.datetime.now()

w = 0.3  #range from 0.3, 0.5, 0.7
c1 = 0.4 #range from 0.4 0.6 0.8
c2 = 0.7 
iteration = 1
particle_num = 20

process_time = pd.read_excel("MOPSO_dataset.xlsx",sheet_name="process_time",index_col =[0])
machine_sequence = pd.read_excel("MOPSO_dataset.xlsx",sheet_name="machines_sequence",index_col =[0])

num_ms = 10 # number of machines
num_job = 20 # number of jobs
particle_size = num_ms * num_job 

pt_array = [list(map(int, process_time.iloc[i])) for i in range(num_job)]
ms_array = [list(map(int, machine_sequence.iloc[i])) for i in range(num_job)]
weight = [1, 10, 1, 5, 10, 10, 5, 10, 5, 1, 5, 1, 1, 10, 1, 1, 10, 5, 5, 1]
due_date = [26, 52, 78, 104, 130, 156, 182, 208, 234, 260, 286, 312, 338, 364, 390, 416, 442, 468, 494, 520]

# 共有兩個目標要衡量-最小化 makesapn & 最小化 total weighted tardiness
# 多目標的用意就是找到兩個目標結果都不會太爛的解
# 一個點是一個 mxn 的一維矩陣

'''initialization'''
p_time = process_time
m_sequence = machine_sequence
particle_array = []
gbest_position = [0 for i in range(particle_size)]
pbest_position = [([0] * particle_size) for i in range(particle_num)]
velocity = [([0] * particle_size) for i in range(particle_num)]

gbest_fitness_value = [99999, 99999]   
pbest_fitness_value = [([99999] * 2) for i in range(particle_num)]
gbest_fitness = 99999
pbest_fitness = [99999 for i in range(particle_num)]

toward_pbest = np.zeros((particle_num,), dtype=int)
toward_gbest = 0

'''init velocity'''
for i in range(particle_num):
    for j in range(particle_size):
        rand_num = random.random()
        if(rand_num <= c2):
            velocity[i][j] = 1
        else:
            velocity[i][j] = 0

'''init particle array'''
for i in range(particle_num):
    particle = list(np.random.permutation(particle_size)) # generate a random permutation of 0 to num_job*num_mc-1
    particle_array.append(particle) # add to the population_list  
    for j in range(particle_size):
        particle_array[i][j] = particle_array[i][j] % num_job # convert to job number format, every job appears m times

'''move particles'''
def move():
    index = np.zeros((10,), dtype=int)
    operation_job = 0
    operation_pbest = 0
    operation_gbest = 0
    
    for i in range(particle_num):
        for j in range(particle_size):
            
            #giffler & thompson algorithm
            job = particle_array[i][j]
            job_pre = particle_array[i][j-1]
            if (due_date[job] < due_date[job_pre]) or (weight[job] > weight[job_pre]):
                particle_array[i][j], particle_array[i][j-1] = particle_array[i][j-1], particle_array[i][j]
            
            m_index = 0
            m_pre_index = 0
            for machines in range(j):
                if (int(particle_array[i][machines]) == job):
                    m_index += 1
                if (int(particle_array[i][machines]) == job_pre):
                    m_pre_index += 1
                
            this_proc_time = int(pt_array[job][m_index - 1])
            pre_proc_time = int(pt_array[job_pre][m_pre_index - 1])
            
            if (this_proc_time < pre_proc_time):
                particle_array[i][j], particle_array[i][j-1] = particle_array[i][j-1], particle_array[i][j]

            lambda_job = particle_array[i][j]
                
            for l in range(j):
                if(particle_array[i][l] == lambda_job):    
                    operation_job += 1
                
            p_or_g = random.random()
            
            if(p_or_g > c1):
                for g in range(particle_size - 10):
                    if(gbest_position[g] == lambda_job):
                        operation_gbest += 1
                        if(gbest_position[g] == lambda_job) and (operation_gbest == operation_job): #compare with gbest 
                            index[0] = g
                            index[1] = g + 1
                            index[2] = g + 2
                            index[3] = g + 3
                            index[4] = g + 4
                            index[5] = g + 5
                            index[6] = g + 6
                            index[7] = g + 7
                            index[8] = g + 8
                            index[9] = g + 9
                            
                            
            else:
               for p in range(particle_size - 10):
                   if(pbest_position[i][p] == lambda_job):
                       operation_pbest += 1
                       if(pbest_position[i][p] == lambda_job) and (operation_pbest == operation_job): #compare with gbest 
                           index[0] = p
                           index[1] = p + 1
                           index[2] = p + 2
                           index[3] = p + 3
                           index[4] = p + 4
                           index[5] = p + 5
                           index[6] = p + 6
                           index[7] = p + 7
                           index[8] = p + 8
                           index[9] = p + 9
    
    for i in range(particle_num):
        for j in range(particle_size - 10):             
            if(velocity[i][j+1] == 0) and (velocity[i][j+2] == 0):
                particle_array[i][index[0]], particle_array[i][j] = particle_array[i][j], particle_array[i][index[0]]
                particle_array[i][index[1]], particle_array[i][j+1] = particle_array[i][j+1], particle_array[i][index[1]]
                particle_array[i][index[2]], particle_array[i][j+2] = particle_array[i][j+2], particle_array[i][index[2]]
                particle_array[i][index[3]], particle_array[i][j+3] = particle_array[i][j+3], particle_array[i][index[3]]
                particle_array[i][index[4]], particle_array[i][j+4] = particle_array[i][j+4], particle_array[i][index[4]]
                particle_array[i][index[5]], particle_array[i][j+5] = particle_array[i][j+5], particle_array[i][index[5]]
                particle_array[i][index[6]], particle_array[i][j+6] = particle_array[i][j+6], particle_array[i][index[6]]
                particle_array[i][index[7]], particle_array[i][j+7] = particle_array[i][j+7], particle_array[i][index[7]]
                particle_array[i][index[8]], particle_array[i][j+8] = particle_array[i][j+8], particle_array[i][index[8]]
                particle_array[i][index[9]], particle_array[i][j+9] = particle_array[i][j+9], particle_array[i][index[9]]
                
                velocity[i][j+9] = 1
            
    return
    
'''calculate fitness'''
def total_fitness():
    #weight * (process time - due date)
    global toward_pbest
    global toward_gbest
    global iteration
           
    for p in range(particle_num):
        job_i = [j for j in range(num_job)]
        count = {key:0 for key in job_i}
        job_count = {key:0 for key in job_i}
        machine_i = [j+1 for j in range(num_ms)]
        machine_count = {key:0 for key in machine_i}
        
        for i in particle_array[p]:
            proc_t = int(pt_array[i][count[i]])
            ma_sq = int(ms_array[i][count[i]])
            job_count[i] = job_count[i] + proc_t
            machine_count[ma_sq] = machine_count[ma_sq] + proc_t
                
            if machine_count[ma_sq] < job_count[i]:
                machine_count[ma_sq] = job_count[i]
            elif machine_count[ma_sq] > job_count[i]:
                job_count[i] = machine_count[ma_sq]            
            count[i] = count[i] + 1
        
        #completion time
        makespan = max(job_count.values()) 
        TWT = 0
        for i in range(num_job):
            tard = job_count[i] - due_date[i]
            weighted_tard = weight[i] * max([0,tard])
            TWT = TWT + weighted_tard
        
        
        if (TWT < pbest_fitness_value[p][0]) or (makespan < pbest_fitness_value[p][1]):
            pbest_fitness_value[p][0] = TWT
            pbest_fitness_value[p][1] = makespan
            toward_pbest[p] = 1
        else:
            toward_pbest[p] = 0
        print("pbest: ", pbest_fitness_value[p])
    
    return

'''choose global best particle using TOPSIS'''
def find_gbest():
    global gbest_fitness_value
    global pbest_fitness_value
    global toward_gbest
    global gbest_fitness
    
    pbest_copy = copy.deepcopy(pbest_fitness_value)
    Z = [([0] * 2) for i in range(particle_num)]
    square_sum_TWT = 0
    square_sum_makespan = 0
    
    for i in range(particle_num):
        square_sum_TWT += pbest_copy[i][0] ** 2
        square_sum_makespan += pbest_copy[i][1] ** 2
    
    sq_root_TWT = np.sqrt(square_sum_TWT)
    sq_root_makespan = np.sqrt(square_sum_makespan)
        
    for i in range(particle_num):
        Z[i][0] = pbest_copy[i][0] / sq_root_TWT
        Z[i][1] = pbest_copy[i][1] / sq_root_makespan
    
        
    Z_best = [min([m[0] for m in Z]), min([m[1] for m in Z])]
    Z_worst = [max([m[0] for m in Z]), max([m[1] for m in Z])]
    
    '''calculate distance'''
    D_best = [0 for i in range(particle_num)]
    D_worst = [0 for i in range(particle_num)]
    for i in range(particle_num):
        D_best[i] = np.sqrt((((Z_best[0] - Z[i][0]) ** 2) + ((Z_best[1] - Z[i][1]) ** 2)) * 0.5)
        D_worst[i] = np.sqrt(((Z_worst[0] - Z[i][0]) ** 2 + (Z_worst[1] - Z[i][1]) ** 2) * 0.5)
    
    ''''計算綜合指標'''
    C = [0 for i in range(particle_num)]
    for i in range(particle_num):
        C[i] = D_worst[i] / (D_best[i] + D_worst[i])
    
    best_particle  = C.index(max(C))
    gbest_fitness = C[best_particle]
    gbest_fitness_value[0] = int(pbest_fitness_value[best_particle][0])
    gbest_fitness_value[1] = int(pbest_fitness_value[best_particle][1])
    
    for i in range(particle_size):
        gbest_position[i] = particle_array[best_particle][i]
    
    if gbest_fitness > 0.9:
        toward_gbest = 1 
    else:
        toward_gbest = 0
    return 
'''main function'''
def move_particles():
    global toward_pbest
    
    global iteration
    global w
    
    for it in range(iteration):
        move()
        total_fitness()
        find_gbest()
        
        for i in range(particle_num):
            tp = int(toward_pbest[i])
            for j in range(particle_size):
                r = random.random()
                if tp == 1 and toward_gbest == 1:
                    velocity[i][j] = 0
                elif tp == 1 and toward_gbest == 0:
                    velocity[i][j] = 1
                elif tp == 0 and toward_gbest == 1:
                    velocity[i][j] = 0
                elif tp == 0 and toward_gbest == 0:
                    velocity[i][j] = 0
                
                if (r <= w):
                    velocity[i][j] = 0
        print("iteration : ", it)
    return

'''main'''
move_particles()
print(gbest_position) 
print(gbest_fitness_value)
print(gbest_fitness)

'''show run time'''
endtime = datetime.datetime.now()
print(endtime - starttime)

'''scatter plot'''
import matplotlib.pyplot as plt

x = []
y = []
for p in range(particle_num):
    x.append(pbest_fitness_value[p][0])
    y.append(pbest_fitness_value[p][1])

plt.scatter(x, y)
plt.xlabel('total weighted tardiness')
plt.ylabel('makespan')
plt.savefig('M-TWT_3_4_7_1000(7).png')


'''plot gantt chart'''
import pandas as pd
import datetime
import matplotlib.pyplot as plt

m_keys = [j+1 for j in range(num_ms)]
j_keys = [j for j in range(num_job)]
key_count = {key:0 for key in j_keys}
j_count = {key:0 for key in j_keys}
m_count = {key:0 for key in m_keys}
record = {}

# Declaring a figure "gnt" 
fig, gnt = plt.subplots() 

# Setting labels for x-axis and y-axis 
gnt.set_xlabel('Process Time') 
gnt.set_ylabel('Machine') 

# Setting Y-axis limits 
gnt.set_ylim(0, 100) 

# Setting ticks on y-axis 
yticks = [(i*10)+5 for i in range(num_ms)]
gnt.set_yticks(yticks) 

# Labelling tickes of y-axis 
yticks_label = [i+1 for i in range(num_ms)]
gnt.set_yticklabels(yticks_label) 

for i in gbest_position:
    gen_t = int(pt_array[i][key_count[i]])
    gen_m = int(ms_array[i][key_count[i]])
    j_count[i] = j_count[i] + gen_t
    m_count[gen_m] = m_count[gen_m] + gen_t
    
    if m_count[gen_m] < j_count[i]:
        m_count[gen_m] = j_count[i]
    elif m_count[gen_m] > j_count[i]:
        j_count[i] = m_count[gen_m]
    
    # convert seconds to hours, minutes and seconds
    start_time = datetime.timedelta(seconds=j_count[i]-pt_array[i][key_count[i]])
    end_time = datetime.timedelta(seconds=j_count[i])
    record[(i,gen_m)] = [start_time.seconds, (end_time - start_time).seconds]
    key_count[i] = key_count[i] + 1

for m in m_keys:
    df_machine = []
    y_min = (m-1) * 10
    for j in j_keys:
        start = record[(j,m)][0]
        duration = record[(j,m)][1]
        df_machine.append((start, duration))
     
    gnt.broken_barh(df_machine, (y_min, 5))
    














