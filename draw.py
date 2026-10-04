import pygame
import os

pygame.init() 

class Canvas:
    def __init__(self, size=(500,500), save_dir='not_square'):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        self.screen.fill((255,255,255))
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.count = len(os.listdir(save_dir))
        self.drawing = False
        self.last_saved = None    

    def save(self):
        path = os.path.join(self.save_dir, f'{self.count}.png')
        pygame.image.save(self.screen, path)
        self.count += 1
        self.screen.fill((255,255,255))
        print(f'saved {path}')
        self.last_saved = path #important as before we had difficulties in accessing the last saved image 
        return path

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.drawing = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.drawing = False
                elif event.type == pygame.MOUSEMOTION and self.drawing:
                    pos = pygame.mouse.get_pos()
                    pygame.draw.circle(self.screen, (0, 0, 0), pos, 4)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    self.save()

            pygame.display.flip()

        pygame.quit()

if  __name__ == '__main__':
    Canvas(save_dir='not_square').run() 
    
