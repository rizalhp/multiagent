import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import timeit


def run(episodes, is_training=True, render=False,):
    
    #my_map = ["SFFF", "FHFF", "FFHF", "FFFG"] #untuk map 4x4
    #my_map = ["SFFFH", "FHFFH", "FFHFF", "HFFFH" ,"FFHFG"] #untuk map 5x5
    my_map = ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF" ,"FFFHFFFG"] #untuk map 8x8
    env = gym.make('FrozenLake-v1', desc = my_map ,map_name="8x8", is_slippery=False, render_mode='human' if render else None)
    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # 8x8 x 4 array
    else:
        f = open('frozen_lake8x8.pkl', 'rb') #WARNING : lihat file yang dibuka karena berpengaruh terhadap kinerja program 
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha atau learning rate
    discount_factor_g = 0.9 # gamma atau discount rate. dimana jika 0 : maka agen lebih mementingkan reward yang datang dalam jangka waktu dekat  sedangkan jika 1 : maka agen akan lebih mementingkan reward yang akan datang dimasa/ jangka waktu panjang 
    epsilon = 1         # 1 = 100% random actions 
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()   # angka acak antara 0 - 1 

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 sampai 63
        terminated = False      # True ketika jatuh ke lubang atau hole atau True ketika mencapai goal
        truncated = False       # True ketika aksi yang diambil > 200
        
 
        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=kiri,1=bawah,2=kanan,3=atas
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)
           
            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action] #rumus Q learning 
                )
            state = new_state
          
        # pengurangan nilai epsilon / esploration rate
        if (epsilon - epsilon_decay_rate == 0):
            epsilon = 0
        else:
            epsilon = epsilon - epsilon_decay_rate
        
        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes) #inisiasi array sum reward dengan 0

    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)]) #pengisian nilai sum reward yang nanti akan di masukan ke dalam plot
    plt.plot(sum_rewards)
    plt.savefig('tess.png')

    #penulisan file pkl untuk menyimpak Q value
    if is_training: 
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()



if __name__ == '__main__':
   
    start = timeit.default_timer()
    run(15000, is_training=True, render=True)
    stop = timeit.default_timer() # catat waktu selesai
    lama_eksekusi = stop - start # lama eksekusi dalam satuan detik
    print("Lama eksekusi: ",lama_eksekusi,"detik")