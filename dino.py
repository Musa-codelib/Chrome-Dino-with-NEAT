import pygame
import random



class dino():
    X_POS=40
    Y_POS=315
    JMP_VEL=8.5
    def __init__(self,img):
        self.image=img
        self.jump_vel=self.JMP_VEL
        self.rect=pygame.Rect(self.X_POS,self.Y_POS,self.image.get_width(),self.image.get_height())
        self.dino_jump=False
        self.dino_run=True
        self.step_index_r=0
        self.step_index_j=0
        self.color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    
    def update(self,JUMPING,RUNNING,game_speed):
        if self.dino_jump==True:
            self.jump(JUMPING,game_speed)
        if self.dino_run==True:
            self.run(RUNNING)
        if self.step_index_r>=40:
            self.step_index_r=0
        if self.step_index_j>=60:
            self.step_index_j=0
    
    def run(self,RUNNING):
        self.image=RUNNING[self.step_index_r//5]
        self.rect.x=self.X_POS
        self.rect.y=self.Y_POS
        self.step_index_r+=5

    def jump(self,JUMPING,game_speed):
        self.image=JUMPING[self.step_index_j//5]
        if self.dino_jump:
            self.rect.y-=self.jump_vel*5+(game_speed*0.01)
            self.jump_vel-=0.9
            self.step_index_j+=5
        if self.jump_vel<=-self.JMP_VEL:
            self.dino_jump=False
            self.dino_run=True
            self.jump_vel=self.JMP_VEL


    def draw(self,screen,obstacles):
        screen.blit(self.image,(self.rect.x,self.rect.y))
        #draw a colored box around our dino
        pygame.draw.rect(screen,self.color,(self.rect.x,self.rect.y,self.rect.width,self.rect.height),2)
        #draw line of sight of dino
        for obs in obstacles:
            pygame.draw.line(screen,self.color,(self.rect.x+34,self.rect.y+12),obs.rect.center,2)