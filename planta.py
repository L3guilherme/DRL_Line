import sys
sys.path.insert(1, './Sim')
from Source import Source
from Queue import Queue
from Machine import Machine
from Exit import Exit
from Failure import Failure
from Repairman import Repairman
from Globals import runSimulation,stepSimulation
import numpy as np
import gym
from gym import spaces
import itertools
import random
import json
from torch.utils.tensorboard import SummaryWriter

class Control_Maquina:
    def __init__(self,machine,v_min,v_max,v_delta,pacote):
        self.machine = machine
        self.vel_min = v_min
        self.vel_max = v_max
        self.delta_vel = v_delta
        self.pacote = pacote

    def Set_Vel(self,val):
        #print(self.machine.name)
        #print('Set: '+str(val))
        #print('Min: '+str(self.vel_min))
        #print('Max: '+str(self.vel_max))
        if(val < self.vel_min):
            #print('Min')
            vel = self.pacote/self.vel_min
        else:
            if (val > self.vel_max):
                #print('Max')
                vel = self.pacote/self.vel_max
            else:
                #print('OK')
                vel = self.pacote/val

        #print('Pross time: '+str(vel))
        #print('Sp: '+str(val)+'/'+str(self.vel_max)+' = '+str(val/self.vel_max))
        var_vel = 0.0005*vel
        min_vel = 0.95*vel
        max_vel = 1.05*vel
            
        self.machine.Set_processingTime(processingTime={'Normal':{'mean':vel,'stdev':var_vel,'min':min_vel,'max':max_vel}})
    def Set_max(self): 
        vel = self.pacote/self.vel_max
        var_vel = 0.0005*vel
        min_vel = 0.95*vel
        max_vel = 1.05*vel
    def Set_min(self): 
        vel = self.pacote/self.vel_min
        var_vel = 0.0005*vel
        min_vel = 0.95*vel
        max_vel = 1.05*vel
            
        self.machine.Set_processingTime(processingTime={'Normal':{'mean':vel,'stdev':var_vel,'min':min_vel,'max':max_vel}})   

class Planta:

    def __init__(self,cfg_file = '',log_dir='plantaLog0',mode = 0,dbug=False):

        super(Planta, self).__init__()

        #print('INIT PLanta...')

        self.delta_sim = 10.0

        self.run1 = True

        self.mode = mode

        self.Load_line(cfg_file)

        self.n_maquinas = len(self.maquinas)

        
        self.acoes = list(itertools.product([0,1,2], repeat = self.n_maquinas))
        self.action_space = spaces.Discrete(len(self.acoes))# N maquinas 3 ações cada
        self.delta_ma_vel = 100.0
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7*self.n_maquinas+self.n_Buffers,), dtype=np.float32)#N maquinas 7 valoes 1 buffer 1 valor
        self.toral_rc = 0.0
        self.itc = 0
        self.itc_total = 0
        self.min_time_check = 500
        self.reward_range = [-np.inf,np.inf]#[-2.0*len(self.maquinas),4.0*len(self.maquinas)]

        self.metadata = {'render.modes': []}

        self.writer = SummaryWriter(log_dir = log_dir)


        self.maxProssTime = -np.inf
        for item in self.objectList:
            try:
                max_at = item.meanProssTime
                if(max_at > self.maxProssTime):
                    self.maxProssTime = max_at
            except Exception as e:
                #print(e.message)
                pass

        if dbug:
            print('------------INIT-----------------')
            print('Procução máxima: '+str(self.pacote_unidades*(1.0/self.maxProssTime))+' cpm')
            print('#Recursos#')
            for mq in self.objectList:
                print('     '+mq.name)
            print('---------------------------------')

        self.th_prod = 0.7

    def Load_line(self,cfgFile ='line.cfg'):
        self.n_Buffers = 0
        self.objectList=[]
        self.maquinas = []
        fObj = open(cfgFile,"r")
        data = json.load(fObj)
        for item in data:
            if(item['Type'] == 'M'):
                M_pt = self.pacote_unidades/float(item['Max'])
                var_M_pt = 0.0005*M_pt
                min_M_pt = 0.95*M_pt
                max_M_pt = 1.05*M_pt
                M=Machine('M'+str(len(self.objectList)),item['Name'], 
                                      processingTime={'Normal':{'mean':M_pt,'stdev':var_M_pt,'min':min_M_pt,'max':max_M_pt}},
                                      min_proprocessingTime =  M_pt)
                mq = Control_Maquina(M,float(item['Min']),float(item['Max']),50,self.pacote_unidades)
                self.maquinas.append(mq)
                self.objectList.append(M)

                R=Repairman('R_'+item['Name'], 'Bob_'+item['Name'])
                ttf = float(item['TTF'])
                ttr = float(item['TTR'])
                ttf_min = ttf*0.5
                ttf_max = ttf*1.5
                ttf_var = 0.05*ttf
                ttr_min = ttr*0.5
                ttr_max = ttr*1.5
                ttr_var = 0.05*ttr
                F=Failure(victim=M, distribution={'TTF':{'Normal':{'mean':ttf,'stdev':ttf_var,'min':ttf_min,'max':ttf_max}},
                                    'TTR':{'Normal':{'mean':ttr,'stdev':ttr_var,'min':ttr_min,'max':ttr_max}}},
                                    repairman=R) 
                self.objectList.append(R)
                self.objectList.append(F)
                    
            if(item['Type'] == 'Q'):
                cap = item['Cap']/self.pacote_unidades
                Q=Queue(id='Q'+str(len(self.objectList)),name=item['Name'],capacity = cap)
                self.objectList.append(Q)
                self.n_Buffers+=1

            if(item['Type'] == 'B'):
                self.pacote_unidades = int(item['batch'])

        E=Exit('E','Exit')
        self.Exit = E
        self.objectList.append(E)

        for item in data:
            eq_pre = []
            eq_pos = []
            eq_at = None
            for pre in item.get('Pre',[]):
                for eq in self.objectList:
                    if(eq.name == pre):
                        eq_pre.append(eq)
            for pos in item.get('Pos',[]):
                for eq in self.objectList:
                    if(eq.name == pos):
                        eq_pos.append(eq)
            for eq in self.objectList:
                if(eq.name == item.get('Name',None)):
                    eq_at = eq

            if(len(eq_pre) == 0 and item.get('Type','N') == 'M'):
                S_pt = 0.9*eq_at.minProssTime
                S=Source('S1_'+eq_at.name,'Source', interArrivalTime={'Fixed':{'mean':S_pt}}, entity='Part')
                eq_pre.append(S)
                S.defineRouting([eq_at])
                self.objectList.append(S)

            if(len(eq_pos) == 0):
                self.Exit.defineRouting([eq_at])
                eq_pos.append(self.Exit)
            if(item.get('Type','N') == 'M' or item.get('Type','N') == 'Q'):
                eq_at.defineRouting(eq_pre,eq_pos)



        
        self.reset()
        #print('OK!')

    def Plot_Var(self,name,value):
        self.writer.add_scalar(f'vars/'+name,value,self.itc_total)


    def Run(self,maxSimTime,seed=42):
        if self.run1 :
            runSimulation(objectList=self.objectList, maxSimTime=maxSimTime,numberOfReplications=1,trace='No', seed=seed)
            self.run1 = False
        else:
            stepSimulation(objectList=self.objectList, maxSimTime=maxSimTime,numberOfReplications=1,trace='No', seed=seed)

        self.itc += 1
        

        self.estado_at = self.Get_res(maxSimTime)
        self.maxProd = self.pacote_unidades*(maxSimTime/self.maxProssTime)
        self.num_Exits = self.pacote_unidades*self.Exit.numOfExits
        self.meanProd = self.pacote_unidades*(self.Exit.numOfExits/maxSimTime)


        #print('------------Saida int '+str(self.itc)+'-----------------')
        #print('Tempo de Sim: '+str(maxSimTime))
        #print('Procução máxima: '+str(self.maxProd)+' unidades')
        #print('Produção: '+str(self.num_Exits)+' unidades = '+str(100*self.num_Exits/self.maxProd)+' %')
        #print('CPM média: '+str(self.meanProd)+' cpm')
        #print('CPM máxima: '+str(self.pacote_unidades*(1.0/self.maxProssTime))+' cpm')
        #print('---------------------------------------------------------')
        self.itc_total += 1



    def Get_res(self,tipo = -1):
        res = []
        for item in self.objectList:
            try:
                res.append(item.Get_Status(tipo))
            except Exception as e:
                #print(e)
                pass
        return res

    def Get_num_estado(self):
        #print(self.estado_at)
        estado = self.estado_at
        vals = []
        vals_mq = []
        for maquina in estado:
            for item in maquina.values():
                vals_tmp = np.fromiter(item.values(), dtype=float)
                vals_mq.append(vals_tmp)
                for val in vals_tmp:
                    vals.append(val)
        #print(vals_mq)
        #print(vals_mq[0])
        return vals,vals_mq

    def Print_Maq_Res(self,ref,maq):
        self.writer.add_scalars(f'Production/'+ref,
                                     {'sp':maq.get('sp',0),
                                      'wp':maq.get('wp',0),
                                      'wt':maq.get('wt',0),
                                      'bp':maq.get('bp',0),
                                      'fp':maq.get('fp',0),
                                      'ot':maq.get('ot',0)}, self.itc_total)

    def Get_RC(self,done):

        rc_maq = []
        for i in range(self.n_maquinas):
            rc_maq.append(0.0)
        if done :
            rc = -100.0
            self.toral_rc += rc
            rc_maq = []
            for i in range(self.n_maquinas):
                rc_maq.append(-10.0)
            return rc,rc_maq

        estado = self.estado_at
        rc = 0.0
        r_bf = 1.0/self.n_Buffers
        total_bf_rw = 0.0
        
        for maquina in estado:  
            for item in maquina:
                wp = maquina[item].get('wp',0)
                wt = maquina[item].get('wt',0)
                bp = maquina[item].get('bp',0)
                sp = maquina[item].get('sp',0)
                ot = maquina[item].get('ot',0)

                rc += wp+sp+ot-wt-bp

                if(item[0] == 'M'):
                    self.Print_Maq_Res(item,maquina[item])
                    index_m = int(item[1:])-1
                    rc_maq[index_m]=(wp+sp+ot-wt-bp)
                if(item[0] == 'Q'):
                    index_b = int(item[1:])-1
                    lvl_bf = maquina[item].get('lv',0)
                    self.writer.add_scalar(f'Buffer/'+item,lvl_bf,self.itc_total)
                    if(lvl_bf > 0.2 and lvl_bf < 0.9):
                        total_bf_rw += +r_bf
                        rc_maq[index_b]+=r_bf
                    else:
                        total_bf_rw += -r_bf
                        rc_maq[index_b]-=r_bf

        rc += total_bf_rw

        if(self.num_Exits < 0.85*self.maxProd): # avaliar punição mais abruptas
            rc += -1.0
        else:
            rc += (self.num_Exits/self.maxProd)

        self.toral_rc += rc
        return rc,rc_maq

        '''prod = (self.num_Exits/self.maxProd)
        #total_wp = total_wp/self.n_maquinas
        #rc = 1.5*prod+1.5*total_wp-1.0
        rc = prod+total_wp
        self.toral_rc += rc
        return rc'''

    def Checar_fim(self):
        if((self.num_Exits < 0.5*self.maxProd) and (self.itc> self.min_time_check)):    
            #print('------------Saida int '+str(self.itc)+'-----------------')
            #print('Procução máxima: '+str(self.maxProd)+' unidades')
            #print('Produção: '+str(self.num_Exits)+' unidades = '+str(100*self.num_Exits/self.maxProd)+' %')
            #print('CPM média: '+str(self.meanProd)+' cpm')
            #print('CPM máxima: '+str(self.pacote_unidades*(1.0/self.maxProssTime))+' cpm')
            #print('Total rc ep: '+str(self.toral_rc))
            #print('---------------------------------------------------------')
            
            return True
        '''else:
            if(self.itc> 2*self.min_time_check):
                print('------------Saida int '+str(self.itc)+'-----------------')
                print('Procução máxima: '+str(self.maxProd)+' unidades')
                print('Produção: '+str(self.num_Exits)+' unidades = '+str(100*self.num_Exits/self.maxProd)+' %')
                print('CPM média: '+str(self.meanProd)+' cpm')
                print('CPM máxima: '+str(self.pacote_unidades*(1.0/self.maxProssTime))+' cpm')
                print('Total rc ep: '+str(self.toral_rc))
                print('---------------------------------------------------------')'''
        return False

    def Exec_action(self, acao):

        if not isinstance(acao,list):
            acao_exec = self.acoes[acao]
        else:
            acao_exec = acao
        #print('-----Acao-------')
        #print('N: '+str(acao))
        #print('Val: '+str(acao_exec))
        #print('----------------------')
        i = 0
        for it in acao_exec:
            vel_n = self.pacote_unidades/self.maquinas[i].machine.totalOperationTimeInCurrentEntity
            if it == 1 :    
                vel_n = vel_n + self.delta_ma_vel
                self.maquinas[i].Set_Vel(vel_n)
            if it == 2 :
                vel_n = vel_n - self.delta_ma_vel
            self.maquinas[i].Set_Vel(vel_n)
            self.writer.add_scalar("Machine Speed/"+self.maquinas[i].machine.name, vel_n,self.itc_total )
            self.Plot_Var('Action_'+str(i),it)
            i+=1


    def step(self,action):
        
        rngSis = random.SystemRandom()
        self.Exec_action(action)
        self.Run(self.IncSimTime+self.delta_sim*(self.itc+1),rngSis.randint(1, 500)) # travar seed
        st_geral,st_mq = self.Get_num_estado()
        estado = np.array(st_geral,dtype=np.float32)
        if(self.num_Exits > self.th_prod*self.maxProd) and (self.itc> 4*self.min_time_check):
            pontos = 100.0
            done = True
            rc_maq = []
            for i_m in range(self.n_maquinas):
                rc_maq.append(10.0)
        else:
            if(self.itc> 8*self.min_time_check):
                pontos = -50.0
                done = True
                rc_maq = []
                for i_m in range(self.n_maquinas):
                    rc_maq.append(-50.0)
                done = True
            else:
                done = self.Checar_fim()
                pontos,rc_maq = self.Get_RC(done)

        perc_prod = 100.0*self.num_Exits/self.maxProd
        self.writer.add_scalar("Reward/Step reward", pontos,self.itc_total )
        self.writer.add_scalar("Reward/Epsode reward", self.toral_rc, self.itc_total )
        self.writer.add_scalar("Production/per", perc_prod, self.itc_total )
        self.writer.add_scalar("Production/abs", self.num_Exits,self.itc_total )
        
        info = perc_prod
        self.writer.flush()
        if self.mode == 1:
            #info = st_mq
            return estado, pontos,rc_maq,done, info
        return estado, pontos,done, info

    def reset(self):
        self.toral_rc = 0
        for mq in self.maquinas:
            mq.Set_min() 
        self.itc = 0
        self.IncSimTime = 1.0
        runSimulation(objectList=self.objectList, maxSimTime=self.IncSimTime,numberOfReplications=1,trace='No', seed=0)
        self.run1 = False
        self.estado_at = self.Get_res(self.IncSimTime)
        st_geral,st_mq = self.Get_num_estado()
        #np.array(self.Get_num_estado(),dtype=np.float32)

        if self.mode == 1:
            return st_geral,st_mq
        return st_geral

    def render(self):
        pass

    def close(self):
        self.writer.close()
