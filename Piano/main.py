from PianoClass import Piano
import time
import pygame

piano = Piano('Piano\PianoNotes\\')

# Ejemplo de Feliz Navidad
song = ['4-b', '4-a', '4-g', '4-a', '4-b', '4-b', '4-b']

for note in song:
    a = note
    piano.PlayNote(a)
    time.sleep(0.5)

# Mantener el programa en ejecución hasta que terminen los audios
while pygame.mixer.get_busy():
    pygame.time.Clock().tick(10)