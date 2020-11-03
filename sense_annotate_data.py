import adagram
import numpy as np
import sys

# OPEN THE ADAGRAM MODEL
path_to_model = '/path/to/model'
model_name = '/model_name'
vm = adagram.VectorModel.load(path_to_model + model_name)

# READ INPUT DATA
path_to_data = '/path/to/data'
filename = sys.argv[1]
data = open(path_to_data + filename, 'r')
lines = [line[:-1].split() for line in data]

# OPEN THE OUTPUT FILE
output = open(sys.argv[-1], 'w')

# DISAMBIGUATE EACH WORD IN A SENTENCE WITH A SLIDING WINDOW
def disambiguate(input):

    sentences = input
    annotated_sents = []
    window_size = 5

    for index, s in enumerate(sentences):
        if len(s) > 0:
            i = 1
            last = s[-1]

            # uncomment print statements to see some details of probabilities and chosen senses
            for ind, w in enumerate(s):
                if w not in vm.dictionary.word2id:
                    # print('\nword \'' + w + '\' not found.\n')
                    if w is last:
                        output.write('\n')
                    if ind > window_size:
                        i += 1

                else:
                    if len(s) < window_size:
                        # print(w, s)
                        probs = vm.disambiguate(w, s)
                        # print('probs:', probs, \
                        # '\nmax prob for \'{}\':'.format(w), max(probs), \
                        # '\nchosen sense index for \'{}\':'.format(w), np.argmax(probs), '\n')
                        new_w = w + str(np.argmax(probs) + 1)
                        if w is last:
                            output.write(new_w + '\n')
                        else:
                            output.write(new_w + ' ')

                    elif ind == 0:
                        context = s[:window_size + 1]
                        # print(w, context)
                        probs = vm.disambiguate(w, context)
                        # print('probs:', probs, \
                        # '\nmax prob for \'{}\':'.format(w), max(probs), \
                        # '\nchosen sense index for \'{}\':'.format(w), np.argmax(probs), '\n')
                        new_w = w + str(np.argmax(probs) + 1)
                        if w is last:
                            output.write(new_w + '\n')
                        else:
                            output.write(new_w + ' ')

                    elif ind < window_size + 1:
                        context = s[:window_size + 1 + ind]
                        # print(w, context)
                        probs = vm.disambiguate(w, context)
                        # print('probs:', probs, \
                        # '\nmax prob for \'{}\':'.format(w), max(probs), \
                        # '\nchosen sense index for \'{}\':'.format(w), np.argmax(probs), '\n')
                        new_w = w + str(np.argmax(probs) + 1)
                        if w is last:
                            output.write(new_w + '\n')
                        else:
                            output.write(new_w + ' ')

                    elif ind > window_size:
                        context = s[i:window_size + 1 + ind]
                        # print(w, context)
                        probs = vm.disambiguate(w, context)
                        # print('probs:', probs, \
                        # '\nmax prob for \'{}\':'.format(w), max(probs), \
                        # '\nchosen sense index for \'{}\':'.format(w), np.argmax(probs), '\n')
                        new_w = w + str(np.argmax(probs) + 1)
                        if w is last:
                            output.write(new_w + '\n')
                        else:
                            output.write(new_w + ' ')

                        i += 1

    return ''

disambiguate(lines)
print()
print('Done!')
output.close()
