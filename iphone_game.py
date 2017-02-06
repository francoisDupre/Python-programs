#!/usr/bin/python
# -*- coding: utf-8 -*-

###############################################################################
#							BEGIN IMPORT 										  #
###############################################################################
import sys
from PyQt4 import QtGui
from random import randint
from PyQt4 import QtCore, QtGui
import math
import copy
import time
###############################################################################
#							END IMPORT 										  #
###############################################################################

###############################################################################
#							BEGIN GLOBAL VARIABLE 							  #
###############################################################################
button_tmp = 0
flag = 0
start = (0,0)
end = (0,0)
board = 0
score = 0
hit = 20
###############################################################################
#							END GLOBAL VARIABLE 							  #
###############################################################################

###############################################################################
#							BEGIN CLASS 									  #
###############################################################################
class Element:
	def __init__(self, name, position):
		self.name = name
		self.position = position
		self.button = 0

class Board:
	def __init__(self):
		self.list_elements = []
		self.list_colors = ['Rouge', 'Jaune', 'Bleu', 
							'Vert', 'Orange', 'Noir',
							'Violet']
		for i in range (7):
			l = []
			for j in range(7):
				l.append(Element('test',(0,0)))
			self.list_elements.append(l)

	def initBoard(self, parent):
		self.initWithoutCorrectNbElement(parent)
		self.changeSameElements()

		for l in self.list_elements:
			for elt in l:
				button = Button(elt.name, parent)
				button.setObjectName('{},{}'.format(elt.position[0], elt.position[1]))
				self.changeColorButton(button, elt.name)				
				elt.button = button

	def changeColorButton(self, button, color):
		if color == self.list_colors[0]:
			button.setStyleSheet("background-color: red")
		if color == self.list_colors[1]:
			button.setStyleSheet("background-color: yellow")
		if color == self.list_colors[2]:
			button.setStyleSheet("background-color: blue")
		if color == self.list_colors[3]:
			button.setStyleSheet("background-color: green")
		if color == self.list_colors[4]:
			button.setStyleSheet("background-color: orange")
		if color == self.list_colors[5]:
			button.setStyleSheet("background-color: black")
		if color == self.list_colors[6]:
			button.setStyleSheet("background-color: purple")
		if color == 'test':
			button.setStyleSheet("background-color: gray")

	def changeSameElements(self):
		conditionNb = self.checkCorrectNbElements(self.list_elements)

		while(conditionNb):
			columns = self.getAlignedElementColumns(self.list_elements)
			for c in columns:
				for l in c:
					if len(l) > 0:
						for elt in l:
							self.changeElement(elt)

			lines = self.getAlignedElementLines(self.list_elements)
			for l in lines:
				for e in l:
					if len(l) > 0:
						for elt in e:
							self.changeElement(elt)
			conditionNb = self.checkCorrectNbElements(self.list_elements)

	def changeElement(self, elt):
		elt = Element('test', (0,0))
		name = self.list_colors[randint(0,6)]
		positionX = randint(0,6)
		positionY = randint(0,6)
		position = (positionX, positionY)
		element = Element(name, position)
		self.list_elements[positionX][positionY] = element

	def changeSameElementsInGame(self):
		conditionNb = self.checkCorrectNbElements(self.list_elements)
		global score
		while(conditionNb):
			columns = self.getAlignedElementColumns(self.list_elements)
			for c in columns:
				for l in c:
					if len(l) > 0:
						score = score + (10 * len(l))
						self.changeElementInGame(l[len(l) - 1], 0, len(l))

			lines = self.getAlignedElementLines(self.list_elements)
			for e in lines:
				for l in e:
					if len(l) > 0:
						score = score + (10 * len(l))
						for elt in l:
							self.changeElementInGame(elt, 1, len(l))
			conditionNb = self.checkCorrectNbElements(self.list_elements)

	
	def changeElementInGame(self, elt, flag, nbElements):

			
		if flag == 1:
			i = 1
			pos = elt.position
			while pos[0] - i >= 0: 
				name = self.list_elements[pos[0] - i][pos[1]].name
				self.changeColorButton(elt.button, name)
				elt.name = name
				elt.button.setText(name)
				elt = self.list_elements[pos[0] - i][pos[1]]
				i = i + 1

			elt = self.list_elements[0][pos[1]]
			elt.name = self.list_colors[randint(0,6)]
			elt.button.setText(elt.name)
			self.changeColorButton(elt.button, elt.name)	

		if flag == 0:
			pos = elt.position
			while pos[0] - nbElements >= 0 and pos[0] - 1 >= 0: 
				name = self.list_elements[pos[0] - nbElements][pos[1]].name
				self.changeColorButton(elt.button, name)
				elt.name = name
				elt.button.setText(name)
				elt = self.list_elements[pos[0] - 1][pos[1]]
				pos = elt.position

			i = nbElements - 1
			while i >= 0:
				elt = self.list_elements[i][pos[1]]
				elt.name = self.list_colors[randint(0,6)]
				elt.button.setText(elt.name)
				self.changeColorButton(elt.button, elt.name)
				i = i - 1
		
	def initWithoutCorrectNbElement(self, parent):
		for i in range (7):
			for j in range (7):
				conditionPos = True
				conditionNb = True
				while conditionPos:
					name = self.list_colors[randint(0,3)]
					positionX = randint(0,6)
					positionY = randint(0,6)
					position = (positionX, positionY)
					conditionPos = self.checkPositionElement(position)
				element = Element(name, position)
				self.list_elements[positionX][positionY] = element

	def checkPositionElement(self, position):
		if self.list_elements[position[0]][position[1]].name != 'test':
			return True
		return False

	def checkCorrectNbElements(self, elements):

		columns = self.getAlignedElementColumns(elements)
		for col in columns:
			for c in col:
				if len(c) > 0:
					return True
		lines = self.getAlignedElementLines(elements)
		for l in lines:
			if len(l) > 0:
				return True

		return False
		
	def getAlignedElementLines(self, elements):
		res = []
		for i in range(7):
			res.append(self.getAlignedElementOneLine(elements[i]))
		return res

	def getAlignedElementColumns(self, elements):
		res = []
		for i in range(7):
			tmp = []
			for j in range(7):
				tmp.append(elements[j][i])
			res.append(self.getAlignedElementOneLine(tmp))
		return res
	def getAlignedElementOneLine(self, line_elements):
		res = []
		res_tmp = []
		i = 0
		while(i < 7):
			res_tmp = []
			elt1 = line_elements[i]
			res_tmp.append(elt1)
			
			j = 1
			if(i + j < 7):
				elt2 = line_elements[i + j]

			while(elt1.name == elt2.name and i + j < 7):
				if(i + j < 7):
					elt2 = line_elements[i + j]
			
				if(elt1.name == elt2.name):
					res_tmp.append(elt2)
				j = j + 1
			
			
			if len(res_tmp) > 2:
				res.append(res_tmp)
				i = i + j - 1
			else:
				i = i + 1	

		return res

	def checkMove(self, start, end):
		if  abs(start[0] - end[0]) > 1:
			return False
		if abs(start[1] - end[1]) > 1 :
			return False
		if start[0] != end[0]:
			if start[1] != end[1]:
				return False

		elt1 = self.list_elements[start[0]][start[1]]
		elt2 = self.list_elements[end[0]][end[1]]

		elts = copy.deepcopy(self.list_elements)

		elts[end[0]][end[1]] = elt1
		elts[start[0]][start[1]] = elt2

		return self.checkCorrectNbElements(elts)

	def validateMove(self, start, end):
		elt1 = self.list_elements[start[0]][start[1]]
		elt2 = self.list_elements[end[0]][end[1]]

		elt1.button.setText(elt2.name)
		elt2.button.setText(elt1.name)

		self.changeColorButton(elt1.button, elt2.name)
		self.changeColorButton(elt2.button, elt1.name)

		self.list_elements[end[0]][end[1]].name = elt2.button.text()
		self.list_elements[start[0]][start[1]].name = elt1.button.text()

		global hit
		hit = hit - 1

	
class GUI(QtGui.QWidget):
	def __init__(self):
		super(GUI, self).__init__()
		self.initUI()

	def initUI(self):
		grid = QtGui.QGridLayout()
		self.setLayout(grid)
 		
		global board
		board = Board()
		board.initBoard(self)
		
		for l in board.list_elements:
			for elt in l:
				button = elt.button
				grid.addWidget(button, *elt.position)


		self.setWindowTitle('My match 3')
		self.show()



class Button(QtGui.QPushButton):
	
	def __init__(self, title, parent):
		super(Button, self).__init__(title, parent)
        
		self.setAcceptDrops(True)

		
	def mouseMoveEvent(self, e):
		if e.buttons() != QtCore.Qt.RightButton:
			return

		mimeData = QtCore.QMimeData()
		mimeData.setText(self.text())

		drag = QtGui.QDrag(self)
		drag.setMimeData(mimeData)

		global button_tmp   
		button_tmp = self


		drag.start(QtCore.Qt.MoveAction)

	def mousePressEvent(self, e):
		super(Button, self).mousePressEvent(e)
		sending_button = self.sender()
		if e.button() == QtCore.Qt.LeftButton:
			print('pressL')
		if e.button() == QtCore.Qt.RightButton:
			print('pressR')
			global flag
			global start
			flag = True
			pos = self.objectName().split(",")
			start = (int(pos[0]), int(pos[1]))

	def dragEnterEvent(self, e):
		e.accept()
		global flag
		pos = self.objectName().split(",")
		end = (int(pos[0]), int(pos[1]))
		if flag and self.text() != button_tmp.text():
			if(board.checkMove(start, end)):
				board.validateMove(start, end)
				board.changeSameElementsInGame()
				printScore()
				flag = False
			else:
				
				flag = False
	
	def mouseReleaseEvent(self,event):  
		if event.button() == QtCore.Qt.RightButton:
			print('release')
			board.changeSameElements()

###############################################################################
#							END CLASS 										  #
###############################################################################

###############################################################################
#							BEGIN FUNCTION									  #
###############################################################################
def main():
	app = QtGui.QApplication(sys.argv)
	ex = GUI()
	sys.exit(app.exec_())

def printScore():
	print('#########################')
	print('Score : {}'.format(score))
	print('Coup restant : {}'.format(hit))
	print('#########################')
	quitApp()

def quitApp():
	if hit < 1:
		QtGui.QApplication.quit()

###############################################################################
#							END FUNCTION									  #
###############################################################################

###############################################################################
#							RUN PROGRAM 									  #
###############################################################################

if __name__ == '__main__':
	main()