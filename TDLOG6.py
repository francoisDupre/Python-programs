class tas_binaire:
	def __init__(self,arbre=[]):
		self.noeuds=arbre
	def add(self,element):
		idx=len(self.noeuds)
		(self.noeuds).append(element)
		while ((idx > 0) and (self.noeuds[idx][0] > self.noeuds[int((idx - 1)/2)][0])):
			a=self.noeuds[idx]
			n=int((idx - 1)/2)
			self.noeuds[idx]=self.noeuds[n]
			self.noeuds[n]=a
			idx=n
	def print_tas(self):
		for i in range(0,len(self.noeuds)):
			print (self.noeuds[i])
	


#arbre2=tas_binaire()	
#arbre2.add([3,"info 1"]) 
#arbre2.add([1,"info 2"]) 
#arbre2.add([9,"info 3"]) 
#arbre2.add([5,"info 4"]) 
#arbre2.add([10,"info 5"]) 

#arbre2.print_tas()


class UnknownInformationException(Exception):
    def __init__(self, informationName):
        super().__init__('Unknown information {}'.format(informationName))
    def __init__(self, informationName):
        super().__init__('Missing information {}'.format(informationName))
    def __init__(self, informationName):
        super().__init__('Missing information {}'.format(informationName))
    def __init__(self, lineNumber, exception):
        self.lineNumber = lineNumber
        self.exception = exception
    def formatError(self):
        return '{}: {}'.format(self.lineNumber + 1, self.exception)
    def __init__(self, sceneErrors):
        self.sceneErrors = sceneErrors
    def printErrors(self):
        for error in self.sceneErrors:
            print(error.formatError())
    def __init__(self):
        self.infos = []
    def load_file(self, filename):
        with open(filename, 'r') as inFile:
            self.load_stream(inFile)
    def load_stream(self, stream):
        errors = []
        for lineNumber, line in enumerate(stream):
            try:
                self.read_line(line)
            except Exception as exception:
                errors.append(SceneError(lineNumber, exception))
        if errors != []:
            raise InvalidSceneException(errors)
        
        words = line.split(":")
        if(len(words) == 2):

            priority = words[0]
            message = words[1].strip()
 
            if priority.isdigit():
                info = []
                info.append(int(priority))
                info.append(message)
                self.infos.append(info)
            else:
                raise UnknownInformationException(priority)
arbre=tas_binaire()
for elt in s.infos:
    arbre.add(elt)

arbre.print_tas()