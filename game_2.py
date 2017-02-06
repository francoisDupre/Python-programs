import numpy as np

def argmax(list):
    l=len(list)
    argMaxs=[0]
    for i in np.arange(1,l):
        if list[i] > list[argMaxs[0]]:
            argMaxs=[i]
        elif (list[i]==list[argMaxs[0]]):
            list.append(i)
    return argMaxs

class Grid:
    def __init__(self,size):
        assert(size%2==1)
        self.size=size
        self.board=np.array((size,size))*np.nan
    def __in__(self,position):
        i,j=(position[0],position[1])# intérêt, vérifier plus tard si on reste dedans
        return (i >=0 and i< self.size and j>= 0 and j< self.size)
    # @property
    # def board(self,i,j):
    #     try:
    #         return self.board[i][j]
    #     except(ValueError):
    #         print("([0),{1)) is out of bounds".format(i,j))
    # def board(self,i,j,value):
    #     assert((i,j) in self.grid)
    #     self.board[i][j]=value
    # @board.setter
    # def board(self,boardValue):
    #     assert(boardValue.shape==(self.board).shape)
    #     self.board=boardValue
    def __str__(self,coinValues,position):
        plusGrandeLongueur=max(map(lambda x:len(str(x)),coinValues))
        boardDisplay=""
        for i in np.arange(self.size):
            boardDisplay+="   ".join([(plusGrandeLongueur-len(str(self.board[i,j])))*" "+str(self.board[i,j])
                                    if position != (i, j)
                                      else plusGrandeLongueur * "#" for j in np.arange(self.size)]) +"\n"
        return boardDisplay

        #
class Player:
    def __init__(self,name="",score=0):
        self.__name=name #bien écouter ce qu'il dit, réfléchir à comment le process
        self.score=score
    def incr(self,value):
        self.score+=value
    def __str__(self):
        return "{0}: {1}".format(self.__name,self.score)


class Game(Grid,Player):
    def __init__(self,size,playerNames,coinValues):
        Grid.__init__(self,size)
        self.numberPlayers=len(playerNames)
        self.players=[Player(playerName) for playerName in playerNames]
        self.numberPlayers=len(playerNames)
        self.coinValues=coinValues
        self.listeDeplacements=dict([(direction, deplacement) for direction, deplacement
                                     in zip(["NW","N","NE","W","E","SW","S","SE"],
                                            [(x,y) for x in [-1,0,1] for y in [-1,0,1] if (x,y) != (0,0)])])
    def gameInitialize(self):
        self.currentPlayer=0
        self.position=((self.size-1)//2,(self.size-1)//2)
        self.board=np.random.choice(self.coinValues,(self.size,self.size))
    def move(self,move):
        newPosition = tuple(map(lambda x, y: x + y, self.position, self.listeDeplacements[move]))
        if newPosition not in self.possibleMoves():
            print("Ce coup n'est pas valable")
        else:
            self.board[self.position]=np.nan
            self.position=newPosition
            self.players[self.currentPlayer].incr(self.board[self.position])
            self.currentPlayer=(self.currentPlayer+1)%self.numberPlayers
    def possibleMoves(self):
        deplacementsPossibles=[]
        for deplacement in self.listeDeplacements.values():
            newPosition = tuple(map(lambda x, y: x + y, self.position, deplacement))
            if (Grid.__in__(self, newPosition)) and (not np.isnan(self.board[newPosition])):
                deplacementsPossibles.append(newPosition)
        return deplacementsPossibles
    def gameOver(self):
        return self.possibleMoves()==[]
    def result(self):
        scores=[player.score for player in self.players]
        # Je n'ai pas géré les égalités
        return argmax(scores)
    def play(self):
        self.gameInitialize()
        while not self.gameOver():
            print("C'est au joueur {0} de jouer".format(self.currentPlayer))
            move=input()
            if move not in ["NW","N","NE","W","E","SW","S","SE"]:
                print("Ce coup n'est pas valable")
                continue
            self.move(move)
        if len(self.result())==1:
            joueurGagnant=self.result()[0]
            print("Le joueur {0} a gagné".format(joueurGagnant))
        else:
            print("Egalité !")
            joueurGagnants=self.result()
            for joueur in joueurGagnants:
                print("Le joueur {0} a gagné".format(joueur))

    def __str__(self):
        # return Grid.__str__(self,self.coinValues,self.position)+Player.__str__(self)
        return Grid.__str__(self, self.coinValues, self.position) + '\n'.join([str(player) for player in self.players])


size=3
playerNames=['Jean','Paul']
coinValues=[5.,10.,20.,50.,100.,200.]
g=Game(size,playerNames,coinValues)
g.play()
print(g)