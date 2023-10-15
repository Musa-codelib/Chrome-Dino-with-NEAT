import pygame


class fireball():
    def __init__(self,img,initial_x_pos,initial_y_pos):
        self.image=img
        self.rect=img.get_rect()
        self.rect.x=initial_x_pos
        self.rect.y=initial_y_pos
        self.step_index=0

    def update(self,obstacles,fireball_imgs,game_speed):
        self.rect.x-=game_speed-(game_speed/4)
        if self.rect.x<-self.rect.width:
            obstacles.pop()
        if self.step_index>=5:
            self.step_index=0
        return obstacles


    def draw(self,screen):
        screen.blit(self.image,self.rect)


