import copy
import os
import time
from collections import namedtuple
import codecs
import numpy as np
# from nltk.align import *

from configs.config import INPUT_SEQUENCE_LENGTH, ANSWER_MAX_TOKEN_LENGTH, TOKEN_REPRESENTATION_SIZE, DATA_PATH, SAMPLES_BATCH_SIZE, \
    TEST_PREDICTIONS_FREQUENCY, TRAIN_BATCH_SIZE, TEST_DATASET_PATH_EN, TEST_DATASET_PATH_DE, NN_MODEL_PATH, FULL_LEARN_ITER_NUM, BUCKETS, PREDICTIONS_FILE
from lib.nn_model.predict import predict_sentence
from lib.w2v_model.vectorizer import get_token_vector
from utilities.utilities import get_logger
from lib.nn_model.model import get_nn_model_new

StatsInfo = namedtuple('StatsInfo', 'start_time, iteration_num, sents_batches_num')

_logger = get_logger(__name__)


def log_predictions(sentences, nn_model, w2v_model, index_to_token, no_predictions, stats_info=None):
    with codecs.open(PREDICTIONS_FILE+'_'+str(no_predictions), 'w', 'utf-8') as predictions:
        for sent in sentences:
            prediction = predict_sentence(sent, nn_model, w2v_model, index_to_token)
            # _logger.info('[%s] -> [%s]' % (sent, prediction))
            print "WRITING PREDICTIONS"
            predictions.write(prediction + '\n')

# def compute_blue_score(sentences_en, sentences_de, nn_model, w2v_model, index_to_token, stats_info=None):
#     bl=[]
#     bl_weights=[0.25, 0.25, 0.25, 0.25]
#     for sent in sentences_en:
#         prediction = predict_sentence(sent, nn_model, w2v_model, index_to_token)
#         # print "BL",bleu(sent,[prediction],bl_weights)
#         bl.append(bleu(sent,[prediction],bl_weights))
#         # _logger.info('[%s] -> [%s]' % (sent, prediction))

#     return (sum(bl)/len(bl))*100


def get_test_senteces(file_path_en, file_path_de):
    with open(file_path_en) as test_data_fh:
        test_sentences_en = test_data_fh.readlines()
        test_sentences_en = [s.strip() for s in test_sentences_en]

    with open(file_path_de) as test_data_fh:
        test_sentences_de = test_data_fh.readlines()
        test_sentences_de = [s.strip() for s in test_sentences_de]

    return test_sentences_en, test_sentences_de


def _batch(tokenized_dialog_lines_en,tokenized_dialog_lines_de, batch_size=2):
    batch = []

    for line_en,line_de in zip(tokenized_dialog_lines_en,tokenized_dialog_lines_de):
        # print "line_en: ", line_en
        # print "line_de: ", line_de
        # print len(line_en)
        batch.append(line_en)
        batch.append(line_de)
        if len(batch) == batch_size:
            yield batch
            batch = []

    # return an empty array instead of yielding incomplete batch
    yield []


def get_training_batch(w2v_model_en, w2v_model_de, tokenized_dialog_en,tokenized_dialog_de, token_to_index_de):
    token_voc_size = len(token_to_index_de)
    for sents_batch in _batch(tokenized_dialog_en,tokenized_dialog_de, SAMPLES_BATCH_SIZE):
        print "sents_batch: ", np.shape(sents_batch)
        if not sents_batch:
            continue

        X = np.zeros((len(sents_batch)/2, INPUT_SEQUENCE_LENGTH, TOKEN_REPRESENTATION_SIZE), dtype=np.float)
        # Y = np.zeros((len(sents_batch)/2, ANSWER_MAX_TOKEN_LENGTH, token_voc_size), dtype=np.bool)
        Y = np.zeros((len(sents_batch)/2, ANSWER_MAX_TOKEN_LENGTH, TOKEN_REPRESENTATION_SIZE), dtype=np.float)
        # for s_index, sentence in enumerate(sents_batch):
        for s_index in range(0, len(sents_batch),2):
            # print "s_index: ",s_index
            # print "s_s_index",s_s_index
            if s_index == len(sents_batch) - 1:
                break

            # print "s_s_index: ",s_index/2
            for t_index, token in enumerate(sents_batch[s_index][:INPUT_SEQUENCE_LENGTH]):
                X[s_index/2, t_index] = get_token_vector(token, w2v_model_en)
                # print "see====>",len(X[s_index/2, t_index])

            for t_index, token in enumerate(sents_batch[s_index + 1][:ANSWER_MAX_TOKEN_LENGTH]):
                # Y[s_index/2, t_index, token_to_index_de[token]] = 1
                Y[s_index/2, t_index] = get_token_vector(token, w2v_model_de)

            # print X[s_index/2]
            # print '-------------------------------------------------'
            # print Y[s_index/2]
            # print "SHAPES X and Y:",np.shape(X),np.shape(Y)

        # print X
        # print '------------'
        # print Y
        yield X, Y


def get_training_batch_new(w2v_model_en, w2v_model_de, tokenized_dialog_en,tokenized_dialog_de, token_to_index_de):
    token_voc_size = len(token_to_index_de)
    for sents_batch in _batch(tokenized_dialog_en,tokenized_dialog_de, SAMPLES_BATCH_SIZE):
        print "sents_batch: ", np.shape(sents_batch)
        if not sents_batch:
            continue

        input_seq_length = len(sents_batch[0])
        output_seq_length = len(sents_batch[1])
        for (a,b) in BUCKETS:
            if a > len(sents_batch[0]):
                input_seq_length = a
                output_seq_length = b
                break

        print "isl", input_seq_length
        print "osl",output_seq_length
        print "len1", sents_batch[0]
        print "len2", sents_batch[1]

        X = np.zeros((len(sents_batch)/2, input_seq_length, TOKEN_REPRESENTATION_SIZE), dtype=np.float)
        Y = np.zeros((len(sents_batch)/2, output_seq_length, token_voc_size), dtype=np.bool)

        nn_model=get_nn_model_new(token_voc_size,input_seq_length, output_seq_length)
        # Y = np.zeros((len(sents_batch)/2, ANSWER_MAX_TOKEN_LENGTH, TOKEN_REPRESENTATION_SIZE), dtype=np.float)
        # for s_index, sentence in enumerate(sents_batch):
        for s_index in range(0, len(sents_batch),2):
            # print "s_index: ",s_index
            # print "s_s_index",s_s_index
            if s_index == len(sents_batch) - 1:
                break

            # print "s_s_index: ",s_index/2
            for t_index, token in enumerate(sents_batch[s_index][:input_seq_length]):
                X[s_index/2, t_index] = get_token_vector(token, w2v_model_en)
                # print "see====>",len(X[s_index/2, t_index])

            for t_index, token in enumerate(sents_batch[s_index + 1][:output_seq_length]):
                Y[s_index/2, t_index, token_to_index_de[token]] = 1
                # Y[s_index/2, t_index] = get_token_vector(token, w2v_model_de)

            # print X[s_index/2]
            # print '-------------------------------------------------'
            # print Y[s_index/2]
            # print "SHAPES X and Y:",np.shape(X),np.shape(Y)

        # print X
        # print '------------'
        # print Y
        yield X, Y,nn_model


def save_model(nn_model):
    # model_full_path = os.path.join(DATA_PATH, 'nn_models', NN_MODEL_PATH)
    model_full_path=NN_MODEL_PATH
    nn_model.save_weights(model_full_path, overwrite=True)



def train_model(nn_model, w2v_model_en, w2v_model_de, tokenized_dialog_lines_en, tokenized_dialog_lines_de, index_to_token_en, index_to_token_de):
    token_to_index_de = dict(zip(index_to_token_de.values(), index_to_token_de.keys()))
    test_sentences_en, test_sentences_de = get_test_senteces(TEST_DATASET_PATH_EN, TEST_DATASET_PATH_DE)

    print "STARTED"

    start_time = time.time()
    no_predictions=0
    sents_batch_iteration = 1

    for full_data_pass_num in xrange(1, FULL_LEARN_ITER_NUM + 1):
        _logger.info('Full-data-pass iteration num: ' + str(full_data_pass_num))
        dialog_lines_for_train_en = copy.copy(tokenized_dialog_lines_en)
        dialog_lines_for_train_de = copy.copy(tokenized_dialog_lines_de)

        for X_train, Y_train in get_training_batch(w2v_model_en, w2v_model_de, dialog_lines_for_train_en, dialog_lines_for_train_de, token_to_index_de):
            nn_model.fit(X_train, Y_train, batch_size=TRAIN_BATCH_SIZE, nb_epoch=10, show_accuracy=True, verbose=1)
            print "SENTECE ITERATION", sents_batch_iteration

            if sents_batch_iteration % TEST_PREDICTIONS_FREQUENCY == 0:
                # print "BLEUUUU"
                # bleu_score = compute_blue_score(test_sentences_en, test_sentences_de, nn_model, w2v_model_en, index_to_token_de)
                log_predictions(test_sentences_en, nn_model, w2v_model_en, index_to_token_de, no_predictions)
                no_predictions+=1
                # print "BLEU SCORE: ", bleu_score 
                save_model(nn_model)

            sents_batch_iteration += 1

        _logger.info('Current time per full-data-pass iteration: %s' % ((time.time() - start_time) / full_data_pass_num))
    save_model(nn_model)




def train_model_no_full_iteration(nn_model, w2v_model_en, w2v_model_de, tokenized_dialog_lines_en, tokenized_dialog_lines_de, index_to_token_en, index_to_token_de):
    token_to_index_de = dict(zip(index_to_token_de.values(), index_to_token_de.keys()))
    test_sentences_en, test_sentences_de = get_test_senteces(TEST_DATASET_PATH_EN, TEST_DATASET_PATH_DE)

    print "STARTED"

    start_time = time.time()
    no_predictions=0
    sents_batch_iteration = 1
    dialog_lines_for_train_en = copy.copy(tokenized_dialog_lines_en)
    dialog_lines_for_train_de = copy.copy(tokenized_dialog_lines_de)
   
    for X_train, Y_train in get_training_batch(w2v_model_en, w2v_model_de, dialog_lines_for_train_en, dialog_lines_for_train_de, token_to_index_de):
        nn_model.fit(X_train, Y_train, batch_size=TRAIN_BATCH_SIZE, nb_epoch=10, show_accuracy=True, verbose=1)
        print "SENTENCE BATCH ITERATION: ",sents_batch_iteration

        if sents_batch_iteration % TEST_PREDICTIONS_FREQUENCY == 0:
    
           
            # bleu_score = compute_blue_score(test_sentences_en, test_sentences_de, nn_model, w2v_model_en, index_to_token_de)
            log_predictions(test_sentences_en, nn_model, w2v_model_en, index_to_token_de, no_predictions)
            no_predictions+=1
            # print "BLEU SCORE: ", bleu_score 
            save_model(nn_model)

        sents_batch_iteration += 1
    save_model(nn_model)


def train_model_new(w2v_model_en, w2v_model_de, tokenized_dialog_lines_en, tokenized_dialog_lines_de, index_to_token_en, index_to_token_de):
    token_to_index_de = dict(zip(index_to_token_de.values(), index_to_token_de.keys()))
    test_sentences_en, test_sentences_de = get_test_senteces(TEST_DATASET_PATH_EN, TEST_DATASET_PATH_DE)

    print "STARTED"

    start_time = time.time()
    sents_batch_iteration = 1

    for full_data_pass_num in xrange(1, FULL_LEARN_ITER_NUM + 1):
        _logger.info('Full-data-pass iteration num: ' + str(full_data_pass_num))
        dialog_lines_for_train_en = copy.copy(tokenized_dialog_lines_en)
        dialog_lines_for_train_de = copy.copy(tokenized_dialog_lines_de)

        for X_train, Y_train,nn_model in get_training_batch_new(w2v_model_en, w2v_model_de, dialog_lines_for_train_en, dialog_lines_for_train_de, token_to_index_de):
            nn_model.fit(X_train, Y_train, batch_size=TRAIN_BATCH_SIZE, nb_epoch=1, show_accuracy=True, verbose=1)
            print "FIT DONE"

            if sents_batch_iteration % TEST_PREDICTIONS_FREQUENCY == 0:
                 print "SENTECE ITERATION", sents_batch_iteration
                # bleu_score = compute_blue_score(test_sentences_en, test_sentences_de, nn_model, w2v_model_en, index_to_token_de)
                log_predictions(test_sentences, nn_model, w2v_model_en, index_to_token_de)
                # print "BLEU SCORE: ", bleu_score 
                save_model(nn_model)

            sents_batch_iteration += 1

        _logger.info('Current time per full-data-pass iteration: %s' % ((time.time() - start_time) / full_data_pass_num))
    save_model(nn_model)
