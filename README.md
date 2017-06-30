#FaceSwapping

##Members

Group 6

Hugo Moraes Dzin - 8532186

Matheus Gomes da Silva Horta - 8532321

Raul Zaninetti Rosa - 8517310

##Abstract

This program will find faces in a picture and replace them. Each recognized face will be compared against a database of famous people's faces, and the one with closest matching facial traits will replace the face found in the picture. It will also be possible to swap around multiple detected faces in the same picture.
Description
Facial recognition is often used by police to identify criminals, in social media to aid in tagging friends in a picture and in applications such as Snapchat to make funny pictures. Our project is oriented towards the third option. The program will take a picture as input, look for faces in it, and replace the matches with different faces. The replacement faces could be the other matches from the same picture, or from a separate database.

Facial recognition will be done using the Viola-Jones algorithm implemented in OpenCV. This algorithm works by applying a sequence of classifiers to regions of the image. When all classifiers succeed, we have found a face (or something very close to one). If any of them fail, the region is considered not be a face and no more classifiers are applied there. Normally, these classifiers need to be trained using a supervised learning model, but we will skip this step by using trained data, in XML format, that comes bundled with OpenCV.

Each identified face will be compared against a database of faces, using parameters such as skin color, texture, and shape of eyes, mouth or nose. The database contains faces of celebrities, ICMC professors and other interesting people. The database entry that most closely matches the identified face will be used to replace it in the picture.

To make the swapped face fit realistically in the original person's head it may need to be resized, have its skin tone altered, and have a smoothing function applied at the boundary between the face and the head.
