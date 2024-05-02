import pygame

# Esta es la clase piano
class Piano:
    def __init__(self, path, pattern='.wav'):
        self.path = path
        self.pattern = pattern
        self.pressed_keys = {}
        # Inicializar Pygame
        pygame.init()

        # Inicializar el mixer de audio para tocar varias notas al mismo tiempo
        pygame.mixer.init()
        
    def PlayNote(self, note):
        if note in self.pressed_keys:
            self.pressed_keys[note].stop()
            del self.pressed_keys[note]
        complete_path = self.path + note + self.pattern
        
        audio = pygame.mixer.Sound(complete_path)
        audio.play()
        
        self.pressed_keys[note] = audio