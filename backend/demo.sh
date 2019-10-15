#Just Testing out some of the Freeman capabilities
echo "Starting Script"

#Create a new project instance, give it basic configurations [ no -c flag necessary if default], train it for 2 epochs on default training method with the john doe data set and save it as demo_1
python3 freeman.py -2 source:john_doe num_epochs:2 -o demo_1

# Now load the project (which will auto save to the same name once finished), train it again, this time using combo training (need to provide positive instances and negative instances)  for 3 epochs
# and then validate an image ~ 7.pgm~
python3 freeman.py  -l demo_1 -3 pos_source:auth_test_pos neg_source:auth_test_neg num_epochs:3 -v 7.pgm


#Create a brand new project, don't train it and save it off to a default name
python3 freeman.py 

#Create a brand new project,train it  set the learning rate to 1 (all other training parameters to default), and save it off to a default name
python3 freeman.py -2 learning_rate:1 

