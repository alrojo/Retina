import getopt, sys







try:
    opts, args = getopt.getopt(sys.argv[1:], "h:o:v", ["help", "output="])
except getopt.GetoptError as err:
    # print help information and exit:
    print(err) # will print something like "option -a not recognized"
    sys.exit(2)

print opts
print args

#print h 
print len(sys.argv)