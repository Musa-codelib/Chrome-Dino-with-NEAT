import pygame

def display_score(points,game_speed,screen,font,speed_increase_sound,position=(0,20)):
    points+=1
    if points%100==0:
        game_speed+=1
       
    text=font.render(f'Points: {str(points)}',True, (0,0,0))
    screen.blit(text,position)
    return points,game_speed

def display_generation(generation,screen,font,position):
    text=font.render(f'Generation No: {str(generation)}',True,(0,0,0))
    screen.blit(text,position)

def display_previous_best_points(prev_best,screen,font,position):
    text=font.render(f'Previous Gen Best Points: {str(prev_best)}',True,(0,0,0))
    screen.blit(text,position)

def display_gamespeed(speed,font,screen,position):
    text=font.render(f'Game Speed: {str(speed)}',True,(0,0,0))
    screen.blit(text,position)

def display_background(x_pos,y_pos,img,game_speed,screen):
    img_w=img.get_width()
    screen.blit(img,(x_pos,y_pos))
    screen.blit(img,(img_w+x_pos,y_pos))
    if x_pos<=-img_w:
        x_pos=0
    x_pos-=game_speed
    return x_pos

def display_total_dinos(total,font,screen,position):
    text=font.render(f'Remaining Dinos: {str(total)}',True,(0,0,0))
    screen.blit(text,position)

def grayscale_image(surf):
    width, height = surf.get_size()
    for x in range(width):
        for y in range(height):
            red, green, blue, alpha = surf.get_at((x, y))
            L = 0.3 * red + 0.59 * green + 0.11 * blue
            gs_color = (L, L, L, alpha)
            surf.set_at((x, y), gs_color)
    return surf