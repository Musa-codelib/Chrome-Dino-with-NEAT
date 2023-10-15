import pygame, sys
from pygame.locals import*
from dino import dino as dinasour
from other import (display_score, display_background,
                    display_gamespeed,grayscale_image,
                    display_total_dinos,display_generation,
                    display_previous_best_points)
from obstacle import  fireball
from NueralNetwork import *
import random


if __name__=="__main__":
    pygame.init()
    icon=pygame.image.load('assets/idle/Idle (1).png')
    pygame.display.set_icon(icon)
    pygame.display.set_caption('DINO game with NEAT algorithm')
    
    #variables required in the game
    default_screen_size=(1250,600)
    screen = pygame.display.set_mode(default_screen_size)
    clock=pygame.time.Clock()
    default_dino_size=(70,60)
    default_fireball_size=(50,50)
    default_saw_size=(55,55)
    obs_spawn_positions=[180,260,315]
    game_speed=20
    game_point=0
    prev_best=game_point
    X_POS_BG=0
    Y_POS_BG=360
    FONT=pygame.font.Font('freesansbold.ttf',20)
    FONT2=pygame.font.Font('freesansbold.ttf',25)
    FONT3=pygame.font.Font('freesansbold.ttf',30)
    jump_sound=pygame.mixer.Sound('assets/sounds/jump.wav')
    speed_increase_sound=pygame.mixer.Sound('assets/sounds/speedincrease.wav')
    ########################################################################
    #loading images
    RUNNING= [pygame.image.load('assets/run/Run (1).png'),
            pygame.image.load('assets/run/Run (2).png'),
            pygame.image.load('assets/run/Run (3).png'),
            pygame.image.load('assets/run/Run (4).png'),
            pygame.image.load('assets/run/Run (5).png'),
            pygame.image.load('assets/run/Run (6).png'),
            pygame.image.load('assets/run/Run (7).png'),
            pygame.image.load('assets/run/Run (8).png')]
    JUMPING= [pygame.image.load('assets/jump/Jump (1).png'),
            pygame.image.load('assets/jump/Jump (2).png'),
            pygame.image.load('assets/jump/Jump (3).png'),
            pygame.image.load('assets/jump/Jump (4).png'),
            pygame.image.load('assets/jump/Jump (5).png'),
            pygame.image.load('assets/jump/Jump (6).png'),
            pygame.image.load('assets/jump/Jump (7).png'),
            pygame.image.load('assets/jump/Jump (8).png'),
            pygame.image.load('assets/jump/Jump (9).png'),
            pygame.image.load('assets/jump/Jump (10).png'),
            pygame.image.load('assets/jump/Jump (11).png'),
            pygame.image.load('assets/jump/Jump (12).png')]
    FIREBALL= [pygame.image.load('assets/hurdles/fireball/red_fireball1.png'),
            pygame.image.load('assets/hurdles/fireball/red_fireball2.png')]
    SAW=[ pygame.image.load('assets/hurdles/Saw.png')]
    BACKGROUND=pygame.image.load('assets/background/Track.png')
    OBSTACLE=[FIREBALL,SAW]
    ########################################################################
    #resizing and grayscaling our images
    for i in range(len(RUNNING)):
        RUNNING[i]=grayscale_image(pygame.transform.scale(RUNNING[i],default_dino_size))
    for i in range(len(JUMPING)):
        JUMPING[i]=grayscale_image(pygame.transform.scale(JUMPING[i],default_dino_size))
    for i in range(len(FIREBALL)):
        FIREBALL[i]=grayscale_image(pygame.transform.scale(FIREBALL[i],default_fireball_size))
    SAW[0]=grayscale_image(pygame.transform.scale(SAW[0],default_saw_size))
    ########################################################################
    #list of dinasours with their nieral networks and fitnesses
        #all the hidden layers and activation layers are specified
        #in the constructor or network class
    dinasours=[dinasour(RUNNING[0]) for i in range(200)]
    dinosours_network=[network(8,1) for i in range(200)]
    #obstacle list initially empty
    obstacles=[fireball(OBSTACLE[random.randint(0,1)][0],default_screen_size[0],obs_spawn_positions[np.random.choice(np.arange(0,3),p=[0.3,0.2,0.5])])]
    ########################################################################
    #function which would return list of 
    #dinos and their NN after selection, repopulation and mutation
    
    
    def repopulate(dino_network):
        t_network=[]
        smallNo=[0.000001,-0.000001]
        for n in dino_network:
            for n2 in dino_network:
                #new networks, crossovered weights would be saved here   
                t_net1=network(8,1)
                t_net2=network(8,1)
                t_net3=network(8,1)
                t_net4=network(8,1)

                t_n1=n.layers[0].weights.flatten().copy()
                t_n2=n.layers[2].weights.flatten().copy()
                t_n3=n.layers[4].weights.flatten().copy()
                t_b1=n.layers[0].bias.flatten().copy()
                t_b2=n.layers[2].bias.flatten().copy()
                t_b3=n.layers[4].bias.flatten().copy()

                t2_n1=n2.layers[0].weights.flatten().copy()
                t2_n2=n2.layers[2].weights.flatten().copy()
                t2_n3=n2.layers[4].weights.flatten().copy()
                t2_b1=n.layers[0].bias.flatten().copy()
                t2_b2=n.layers[2].bias.flatten().copy()
                t2_b3=n.layers[4].bias.flatten().copy()

                #crossover between
                #genome1=A1-B1
                #genome2=A2-B2

                #crossover would result in
                #A2-B1
                #A1-B2
                temp=t_n1[:40].copy()
                t_n1[:40],t2_n1[:40]=t2_n1[:40],temp
                temp2=t_n2[:30].copy()
                t_n2[:30],t2_n2[:30]=t2_n2[:30],temp2
                temp3=t_n3[:3].copy()
                t_n3[:3],t2_n3[:3]=t2_n3[:3],temp3
                temp4=t_b1[:5].copy()#bias FClayer1
                t_b1[:5],t2_b1[:5]=t2_b1[:5],temp4
                temp5=t_b2[:3].copy()#bias FClayer2
                t_b2[:3],t2_b2[:3]=t2_b2[:3],temp5
    


                #converting back to 2D array
                t_net1.layers[0].weights=np.reshape(t_n1,(8,10))
                t_net1.layers[2].weights=np.reshape(t_n2,(10,6))
                t_net1.layers[4].weights=np.reshape(t_n3,(6,1))
                t_net1.layers[0].bias=np.reshape(t_b1,(1,10))
                t_net1.layers[2].bias=np.reshape(t_b2,(1,6))
                t_net1.layers[4].bias=np.reshape(t_b3,(1,1))

                t_net2.layers[0].weights=np.reshape(t2_n1,(8,10))
                t_net2.layers[2].weights=np.reshape(t2_n2,(10,6))
                t_net2.layers[4].weights=np.reshape(t2_n3,(6,1))
                t_net2.layers[0].bias=np.reshape(t2_b1,(1,10))
                t_net2.layers[2].bias=np.reshape(t2_b2,(1,6))
                t_net2.layers[4].bias=np.reshape(t2_b3,(1,1))

                #mutation
                #----in weights
                t_net1.layers[0].weights[random.randint(0,7)][5:]=np.random.rand(1,5).flatten()-0.5
                t_net1.layers[2].weights[random.randint(0,9)][3:]=np.random.rand(1,3).flatten()-0.5
                t_net1.layers[4].weights[random.randint(0,5)][0]=random.random()-0.5

                t_net2.layers[0].weights[random.randint(0,7)]=np.random.rand(1,10).flatten()-0.5
                t_net2.layers[2].weights[random.randint(0,9)]=np.random.rand(1,6).flatten()-0.5
                t_net2.layers[4].weights[random.randint(0,5)][0]=random.random()-0.5
                #---in bias
                t_net1.layers[0].bias[0][random.randint(0,9)]=random.random()-0.5
                t_net1.layers[2].bias[0][random.randint(0,5)]=random.random()-0.5

                t_net2.layers[0].bias[0][random.randint(0,9)]=random.random()-0.5
                t_net2.layers[2].bias[0][random.randint(0,5)]=random.random()-0.5

                t_network.append(t_net1)
                t_network.append(t_net2)

                
                t_n1=n.layers[0].weights.flatten().copy()
                t_n2=n.layers[2].weights.flatten().copy()
                t_n3=n.layers[4].weights.flatten().copy()
                t_b1=n.layers[0].bias.flatten().copy()
                t_b2=n.layers[2].bias.flatten().copy()
                t_b3=n.layers[4].bias.flatten().copy()

                t2_n1=n2.layers[0].weights.flatten().copy()
                t2_n2=n2.layers[2].weights.flatten().copy()
                t2_n3=n2.layers[4].weights.flatten().copy()
                t2_b1=n.layers[0].bias.flatten().copy()
                t2_b2=n.layers[2].bias.flatten().copy()
                t2_b3=n.layers[4].bias.flatten().copy()
                
                
                 #crossover would result in
                #A1-B2
                #A2-B1
                #same result as above but mutation would be different 
                temp=t_n1[40:].copy()
                t_n1[40:],t2_n1[40:]=t2_n1[40:],temp
                temp2=t_n2[30:].copy()
                t_n2[30:],t2_n2[30:]=t2_n2[30:],temp2
                temp3=t_n3[3:].copy()
                t_n3[3:],t2_n3[3:]=t2_n3[3:],temp3
                temp4=t_b1[5:].copy()#bias FClayer1
                t_b1[5:],t2_b1[5:]=t2_b1[5:],temp4
                temp5=t_b2[3:].copy()#bias FClayer2
                t_b2[3:],t2_b2[3:]=t2_b2[3:],temp5

                #converting back to 2D array
                t_net3.layers[0].weights=np.reshape(t_n1,(8,10))
                t_net3.layers[2].weights=np.reshape(t_n2,(10,6))
                t_net3.layers[4].weights=np.reshape(t_n3,(6,1))
                t_net3.layers[0].bias=np.reshape(t_b1,(1,10))
                t_net3.layers[2].bias=np.reshape(t_b2,(1,6))
                t_net3.layers[4].bias=np.reshape(t_b3,(1,1))

                t_net4.layers[0].weights=np.reshape(t2_n1,(8,10))
                t_net4.layers[2].weights=np.reshape(t2_n2,(10,6))
                t_net4.layers[4].weights=np.reshape(t2_n3,(6,1))
                t_net4.layers[0].bias=np.reshape(t2_b1,(1,10))
                t_net4.layers[2].bias=np.reshape(t2_b2,(1,6))
                t_net4.layers[4].bias=np.reshape(t2_b3,(1,1))

                #mutation
                #----in weights
                t_net3.layers[0].weights[random.randint(0,7)][:5]=np.random.rand(1,5).flatten()-0.5
                t_net3.layers[2].weights[random.randint(0,9)][:3]=np.random.rand(1,3).flatten()-0.5
                t_net3.layers[4].weights[random.randint(0,5)][0]=random.random()-0.5
                
                t_net4.layers[0].weights[random.randint(0,7)]=np.random.rand(1,10).flatten()-0.5
                t_net4.layers[2].weights[random.randint(0,9)]=np.random.rand(1,6).flatten()-0.5
                
                #---in bias
                t_net3.layers[0].bias[0][random.randint(0,9)]=random.random()-0.5
                t_net3.layers[2].bias[0][random.randint(0,5)]=random.random()-0.5
                t_net3.layers[4].bias[0][0]+=smallNo[random.randint(0,1)]
                t_net4.layers[0].bias[0][random.randint(0,9)]=random.random()-0.5
                t_net4.layers[2].bias[0][random.randint(0,5)]=random.random()-0.5
                
                t_network.append(t_net3)
                t_network.append(t_net4)
        return t_network


    temp_nets=[]
    Generation=0
    # main game loop
    while True:
        if len(dinasours)==0:
            dinasours=[dinasour(RUNNING[0]) for i in range(200)]
            dinosours_network = repopulate(temp_nets[193:])
            dinosours_network.extend(temp_nets[196:])
            #print("Total networks: ",len(dinosours_network))
            game_speed=20
            prev_best=game_point
            game_point=0
            Generation+=1
            obstacles.clear()
            temp_nets.clear()
        #To exit the game once user clicks quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        #coloring the screen white initially
        screen.fill((255,255,255))

        #displaying background and increasing gamespeed after each 10 points
        X_POS_BG=display_background(X_POS_BG,Y_POS_BG,BACKGROUND,game_speed,screen)
        game_point,game_speed=display_score(game_point,game_speed,screen,FONT2,speed_increase_sound,(screen.get_width()/2-40,20))
        display_previous_best_points(prev_best,screen,FONT,(10,460))
        #displaying game speed ad total dinos remaining
        display_gamespeed(game_speed-20,FONT,screen,(10,400))
        display_total_dinos(len(dinasours),FONT,screen,(10,430))
        display_generation(Generation,screen,FONT3,(screen.get_width()/2-80,screen.get_height()-30))

        #adding a new obstacle once previous one has passed
        if len(obstacles)==0:
            obstacles.append(fireball(OBSTACLE[random.randint(0,1)][0],default_screen_size[0],obs_spawn_positions[np.random.choice(np.arange(0,3),p=[0.3,0.2,0.5])]))

        #displaying dinos
        for dino in dinasours:
            dino.update(JUMPING,RUNNING,game_speed)
            dino.draw(screen,obstacles)

        
        #feeding our nn inputs and predicting either dino should jump or not
        #inputs to our nueral network are:
        #1)distance between player and obstacle
        #2)game speed
        #3)obstacle width
        #4)obstacle heigh
        #5)x position of our obstacle
        #6)Y position of our obstacle
        #7)X position of our dino
        #8)Y position of our dino
        for i, dino in enumerate(dinasours):
            #fitness[i]+=game_point
            dist=abs(obstacles[0].rect.x-dino.rect.x) #distance between obs and dino
            pred=dinosours_network[i].predict(np.array([dist/10,game_speed*2,obstacles[0].rect.width,obstacles[0].rect.height,obstacles[0].rect.x/10,obstacles[0].rect.y,dino.rect.x/10,dino.rect.y]))
            #print(pred)
            print(pred)
            if  pred>=0.5: #jump if P(jump)>=0.5
                dino.dino_jump=True
                dino.dino_run=False

        #removing dinosour from our list once it has collided with the obstacle
        for obs in obstacles:
            obs.draw(screen)
            obstacles=obs.update(obstacles,OBSTACLE[random.randint(0,1)],game_speed)
            for i,dino in enumerate(dinasours):
                if dino.rect.colliderect(obs.rect):
                    temp_nets.append(dinosours_network.pop(i))
                    dinasours.pop(i)
                    
        clock.tick(40)
        pygame.display.update()
