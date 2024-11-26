import nltk
import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
import random
import string
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure necessary NLTK corpora are downloaded
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

class MarkovChain:
    def __init__(self):
        self.chain = {}
        self.start_states = []

    def train(self, sequences):
        for seq in sequences:
            if seq:
                self.start_states.append(seq[0])
            for i in range(len(seq) - 1):
                current_state = seq[i]
                next_state = seq[i + 1]
                if current_state not in self.chain:
                    self.chain[current_state] = {}
                if next_state not in self.chain[current_state]:
                    self.chain[current_state][next_state] = 0
                self.chain[current_state][next_state] += 1

    def predict_next(self, current_state):
        if current_state in self.chain:
            next_states = self.chain[current_state]
            total = sum(next_states.values())
            rand_val = random.uniform(0, total)
            cumulative = 0
            for state, count in next_states.items():
                cumulative += count
                if rand_val <= cumulative:
                    return state
        return None

    def get_start_state(self):
        return random.choice(self.start_states) if self.start_states else None

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return None

def extract_subject_predicate(sentence):
    words = word_tokenize(sentence)
    words = [word for word in words if word not in string.punctuation]
    pos_tags = pos_tag(words)

    subjects = []
    predicates = []
    current_subject = []
    current_verb_phrase = []
    first_verb_index = None
    conjunction = False

    for i, (word, tag) in enumerate(pos_tags):
        wn_pos = get_wordnet_pos(tag)

        if tag == "DT" and (i + 1 < len(pos_tags) and pos_tags[i + 1][1].startswith("NN")):
            current_subject.append(word)

        if wn_pos == wordnet.NOUN or tag == "PRP":
            if first_verb_index is None or conjunction:
                current_subject.append(word)
                conjunction = False

        if wn_pos == wordnet.VERB or tag.startswith("VB") or tag in ["MD", "TO", "VBG", "VBN"]:
            if current_subject:
                subjects.append(" ".join(current_subject))
                current_subject = []
            current_verb_phrase.append(word)
            first_verb_index = i
        else:
            if current_verb_phrase:
                predicates.append(" ".join(current_verb_phrase))
                current_verb_phrase = []

        if tag == "CC":
            conjunction = True

    if current_subject:
        subjects.append(" ".join(current_subject))
    if current_verb_phrase:
        predicates.append(" ".join(current_verb_phrase))

    return subjects, predicates

def predict_subject_predicate_markov(words, markov_chain):
    subjects = []
    predicates = []
    current_subject = []
    current_predicate = []
    predicted_pos = markov_chain.get_start_state()

    for i, word in enumerate(words):
        if word in string.punctuation:
            continue
        if i == 0:
            pos = predicted_pos
        else:
            pos = markov_chain.predict_next(predicted_pos)
            if pos is None:
                pos = predicted_pos
        predicted_pos = pos

        if pos == wordnet.NOUN:
            current_subject.append(word)
            if current_predicate:
                predicates.append(" ".join(current_predicate))
                current_predicate = []
        elif pos == wordnet.VERB:
            current_predicate.append(word)
            if current_subject:
                subjects.append(" ".join(current_subject))
                current_subject = []
        else:
            if current_subject:
                subjects.append(" ".join(current_subject))
                current_subject = []
            if current_predicate:
                predicates.append(" ".join(current_predicate))
                current_predicate = []

    if current_subject:
        subjects.append(" ".join(current_subject))
    if current_predicate:
        predicates.append(" ".join(current_predicate))

    return subjects, predicates

def get_label_sequence(sentence, subjects, predicates):
    words = word_tokenize(sentence)
    labels = ["Other"] * len(words)

    for label_type, phrases in [
        ("Subject", subjects),
        ("Predicate", predicates),
    ]:
        if pd.isna(phrases):
            continue
        for phrase in phrases.split(", "):
            phrase_words = word_tokenize(phrase)
            phrase_len = len(phrase_words)
            for i in range(len(words) - phrase_len + 1):
                if words[i : i + phrase_len] == phrase_words:
                    for j in range(phrase_len):
                        labels[i + j] = label_type

    return words, labels

def get_label_sequence_from_phrases(words, subjects, predicates):
    labels = ["Other"] * len(words)
    for label_type, phrases in [
        ("Subject", subjects),
        ("Predicate", predicates),
    ]:
        for phrase in phrases:
            phrase_words = word_tokenize(phrase)
            phrase_len = len(phrase_words)
            for i in range(len(words) - phrase_len + 1):
                if words[i:i+phrase_len] == phrase_words:
                    for j in range(phrase_len):
                        labels[i + j] = label_type
    return labels

def predict_labels_markov(words, markov_chain_labels):
    labels = []
    current_label = markov_chain_labels.get_start_state()
    for i, word in enumerate(words):
        if word in string.punctuation:
            labels.append("Other")
            continue
        if i == 0:
            label = current_label
        else:
            label = markov_chain_labels.predict_next(current_label)
            if label is None:
                label = current_label
        labels.append(label)
        current_label = label
    return labels

def extract_subject_predicate_labels(words, labels):
    subjects = []
    predicates = []
    current_subject = []
    current_predicate = []
    for word, label in zip(words, labels):
        if word in string.punctuation:
            continue
        if label == "Subject":
            current_subject.append(word)
            if current_predicate:
                predicates.append(" ".join(current_predicate))
                current_predicate = []
        elif label == "Predicate":
            current_predicate.append(word)
            if current_subject:
                subjects.append(" ".join(current_subject))
                current_subject = []
        else:
            if current_subject:
                subjects.append(" ".join(current_subject))
                current_subject = []
            if current_predicate:
                predicates.append(" ".join(current_predicate))
                current_predicate = []
    if current_subject:
        subjects.append(" ".join(current_subject))
    if current_predicate:
        predicates.append(" ".join(current_predicate))
    return subjects, predicates

def calculate_accuracy(actual_labels, predicted_labels):
    subject_correct = 0
    subject_total = 0
    predicate_correct = 0
    predicate_total = 0

    for a_label, p_label in zip(actual_labels, predicted_labels):
        if a_label == 'Subject':
            subject_total += 1
            if p_label == 'Subject':
                subject_correct += 1
        elif a_label == 'Predicate':
            predicate_total += 1
            if p_label == 'Predicate':
                predicate_correct += 1

    subject_accuracy = (subject_correct / subject_total * 100) if subject_total > 0 else 0
    predicate_accuracy = (predicate_correct / predicate_total * 100) if predicate_total > 0 else 0
    return subject_accuracy, predicate_accuracy

def plot_accuracy(accuracies):
    methods = ['Actual POS', 'Markov POS', 'Markov XLSX POS']
    subject_accuracies = [acc[0] for acc in accuracies]
    predicate_accuracies = [acc[1] for acc in accuracies]

    x = range(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x, subject_accuracies, width, label='Subject Accuracy')
    rects2 = ax.bar([i + width for i in x], predicate_accuracies, width, label='Predicate Accuracy')

    for i, rect in enumerate(rects1):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                f'{subject_accuracies[i]:.2f}%',
                ha='center', va='bottom')

    for i, rect in enumerate(rects2):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                f'{predicate_accuracies[i]:.2f}%',
                ha='center', va='bottom')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Subject and Predicate Prediction Accuracy')
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 110)  # Set y-axis limit to show percentages properly
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Read data from the XLSX file
    df = pd.read_excel("combined_updated_labeled_sentences.xlsx")

    # Split data into training and testing sets
    train_df = df[:-35]
    test_df = df[-35:]

    # Process training data for Markov Chain labels
    label_sequences = []
    for idx, row in train_df.iterrows():
        sentence = row["Sentence"]
        subjects = row["Subject"]
        predicates = row["Predicate"]
        words, labels = get_label_sequence(sentence, subjects, predicates)
        label_sequences.append(labels)

    # Initialize and train Markov Chain for labels
    markov_chain_labels = MarkovChain()
    markov_chain_labels.train(label_sequences)

    # Prepare training data for POS Markov Chain from the Sentence column
    training_sentences_pos = train_df["Sentence"].tolist()

    # Initialize and train Markov Chain for POS tags
    markov_chain_pos = MarkovChain()
    markov_chain_pos.train(
        [
            [
                get_wordnet_pos(tag)
                for _, tag in pos_tag(word_tokenize(sentence))
                if get_wordnet_pos(tag) is not None
            ]
            for sentence in training_sentences_pos
        ]
    )

    # Test the model and compare results
    methods = ['Actual POS', 'Markov POS', 'Markov XLSX POS']
    accuracies = {method: {'Subject': [], 'Predicate': []} for method in methods}

    for idx, row in test_df.iterrows():
        sentence = row["Sentence"]
        actual_subjects = row["Subject"]
        actual_predicates = row["Predicate"]
        words = word_tokenize(sentence)

        # Get the actual labels from the annotations
        words, actual_labels = get_label_sequence(sentence, actual_subjects, actual_predicates)

        # Method 1: Actual POS
        extracted_actual_subjects, extracted_actual_predicates = extract_subject_predicate(sentence)
        extracted_labels = get_label_sequence_from_phrases(words, extracted_actual_subjects, extracted_actual_predicates)

        # Method 2: Markov POS
        markov_subjects, markov_predicates = predict_subject_predicate_markov(words, markov_chain_pos)
        markov_pos_labels = get_label_sequence_from_phrases(words, markov_subjects, markov_predicates)

        # Method 3: Markov XLSX POS
        predicted_labels = predict_labels_markov(words, markov_chain_labels)

        # Calculate accuracies
        actual_pos_subject_accuracy, actual_pos_predicate_accuracy = calculate_accuracy(actual_labels, extracted_labels)
        markov_pos_subject_accuracy, markov_pos_predicate_accuracy = calculate_accuracy(actual_labels, markov_pos_labels)
        markov_xlsx_subject_accuracy, markov_xlsx_predicate_accuracy = calculate_accuracy(actual_labels, predicted_labels)

        # Collect accuracies
        accuracies['Actual POS']['Subject'].append(actual_pos_subject_accuracy)
        accuracies['Actual POS']['Predicate'].append(actual_pos_predicate_accuracy)
        accuracies['Markov POS']['Subject'].append(markov_pos_subject_accuracy)
        accuracies['Markov POS']['Predicate'].append(markov_pos_predicate_accuracy)
        accuracies['Markov XLSX POS']['Subject'].append(markov_xlsx_subject_accuracy)
        accuracies['Markov XLSX POS']['Predicate'].append(markov_xlsx_predicate_accuracy)

        # Output the results for each sentence
        print(f"Sentence: {sentence}")
        print(
            f"Actual POS - Subjects: {', '.join(extracted_actual_subjects) if extracted_actual_subjects else 'None'}, Predicates: {', '.join(extracted_actual_predicates) if extracted_actual_predicates else 'None'}"
        )
        print(
            f"Markov POS - Subjects: {', '.join(markov_subjects) if markov_subjects else 'None'}, Predicates: {', '.join(markov_predicates) if markov_predicates else 'None'}"
        )
        print(
            f"Markov XLSX POS - Subjects: {', '.join(extract_subject_predicate_labels(words, predicted_labels)[0]) if extract_subject_predicate_labels(words, predicted_labels)[0] else 'None'}, Predicates: {', '.join(extract_subject_predicate_labels(words, predicted_labels)[1]) if extract_subject_predicate_labels(words, predicted_labels)[1] else 'None'}\n"
        )

    # Calculate average accuracies
    average_accuracies = []
    for method in methods:
        subject_acc_list = accuracies[method]['Subject']
        predicate_acc_list = accuracies[method]['Predicate']
        subject_accuracy = sum(subject_acc_list) / len(subject_acc_list) if len(subject_acc_list) > 0 else 0
        predicate_accuracy = sum(predicate_acc_list) / len(predicate_acc_list) if len(predicate_acc_list) > 0 else 0
        average_accuracies.append((subject_accuracy, predicate_accuracy))

    # Plot accuracies
    plot_accuracy(average_accuracies)
