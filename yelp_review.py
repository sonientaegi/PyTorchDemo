import json
import string
import time
from argparse import Namespace
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# args = Namespace(
#     raw_train_dataset_csv="data/yelp/raw_train.csv",
#     raw_test_dataset_csv="data/yelp/raw_test.csv",
#     proportion_subset_of_train=0.1,
#     train_proportion=0.7,
#     val_proportion=0.15,
#     test_proportion=0.15,
#     output_munged_csv="data/yelp/reviews_with_splits_lite.csv",
#     seed=1337
# )
#
# # 원본 데이터를 읽습니다
# train_reviews = pd.read_csv(args.raw_train_dataset_csv, header=None, names=['rating', 'review'])
#
# # 리뷰 클래스 비율이 동일하도록 만듭니다
# by_rating = collections.defaultdict(list)
# for _, row in train_reviews.iterrows():
#     by_rating[row.rating].append(row.to_dict())
#
# review_subset = []
#
# for _, item_list in sorted(by_rating.items()):
#
#     n_total = len(item_list)
#     n_subset = int(args.proportion_subset_of_train * n_total)
#     review_subset.extend(item_list[:n_subset])
#
# review_subset = pd.DataFrame(review_subset)
#
# # 훈련, 검증, 테스트를 만들기 위해 별점을 기준으로 나눕니다
# by_rating = collections.defaultdict(list)
# for _, row in review_subset.iterrows():
#     by_rating[row.rating].append(row.to_dict())
#
# # 분할 데이터를 만듭니다.
# final_list = []
# np.random.seed(args.seed)
#
# for _, item_list in sorted(by_rating.items()):
#
#     np.random.shuffle(item_list)
#
#     n_total = len(item_list)
#     n_train = int(args.train_proportion * n_total)
#     n_val = int(args.val_proportion * n_total)
#     n_test = int(args.test_proportion * n_total)
#
#     # 데이터 포인터에 분할 속성을 추가합니다
#     for item in item_list[:n_train]:
#         item['split'] = 'train'
#
#     for item in item_list[n_train:n_train+n_val]:
#         item['split'] = 'val'
#
#     for item in item_list[n_train+n_val:n_train+n_val+n_test]:
#         item['split'] = 'test'
#
#     # 최종 리스트에 추가합니다
#     final_list.extend(item_list)
#
# final_reviews = pd.DataFrame(final_list)
#
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'([.,!?])', r' \1 ', text)
#     text = re.sub(r'[^a-zA-Z.,!?]+', r' ', text)
#     return text
#
# final_reviews.review = final_reviews.review.apply(preprocess_text)

args = Namespace(
    # 날짜와 경로 정보
    frequency_cutoff=25,
    model_state_file='model.mlp.pth',
    review_csv='data/yelp/reviews_with_splits_lite.csv',
    # review_csv='data/yelp/reviews_with_splits_full.csv',
    save_dir='model_storage/ch3/yelp/',
    vectorizer_file='vectorizer.json',
    # 모델 하이퍼파라미터 없음
    # 훈련 하이퍼파라미터
    batch_size=128,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=20,
    seed=1337,
    # 실행 옵션
    catch_keyboard_interrupt=True,
    device="mps",
    expand_filepaths_to_save_dir=True,
    reload_from_files=False,
    num_hidden_features=4096
)
np.random.seed(args.seed)

class ReviewDataset(Dataset):
    def __init__(self, review_df, vectorizer):
        """
        매개변수:
            review_df (pandas.DataFrame): 데이터셋
            vectorizer (ReviewVectorizer): ReviewVectorizer 객체
        """
        self.review_df = review_df
        self._vectorizer = vectorizer

        self.train_df = self.review_df[self.review_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.review_df[self.review_df.split=='val']
        self.validation_size = len(self.val_df)

        self.test_df = self.review_df[self.review_df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv):
        """ 데이터셋을 로드하고 새로운 ReviewVectorizer 객체를 만듭니다

        매개변수:
            review_csv (str): 데이터셋의 위치
        반환값:
            ReviewDataset의 인스턴스
        """
        review_df = pd.read_csv(review_csv)
        return cls(review_df, ReviewVectorizer.from_dataframe(review_df))

    def get_vectorizer(self):
        """ 벡터 변환 객체를 반환합니다 """
        return self._vectorizer

    def set_split(self, split="train"):
        """ 데이터프레임에 있는 열을 사용해 분할 세트를 선택합니다

        매개변수:
            split (str): "train", "val", "test" 중 하나
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """ 파이토치 데이터셋의 주요 진입 메서드

        매개변수:
            index (int): 데이터 포인트의 인덱스
        반환값:
            데이터 포인트의 특성(x_data)과 레이블(y_target)로 이루어진 딕셔너리
        """
        row = self._target_df.iloc[index]
        review_vector = self._vectorizer.vectorize(row.review)
        rating_index = self._vectorizer.rating_vocab.lookup_token(row.rating)

        return {'x_data': review_vector, 'y_target': rating_index}

    def get_num_batches(self, batch_size):
        """ 배치 크기가 주어지면 데이터셋으로 만들 수 있는 배치 개수를 반환합니다

        매개변수:
            batch_size (int)
        반환값:
            배치 개수
        """
        return len(self) // batch_size


class Vocabulary(object):
    """ 매핑을 위해 텍스트를 처리하고 어휘 사전을 만드는 클래스 """

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        매개변수:
            token_to_idx (dict): 기존 토큰-인덱스 매핑 딕셔너리
            add_unk (bool): UNK 토큰을 추가할지 지정하는 플래그
            unk_token (str): Vocabulary에 추가할 UNK 토큰
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)


    def to_serializable(self):
        """ 직렬화할 수 있는 딕셔너리를 반환합니다 """
        return {'token_to_idx': self._token_to_idx,
                'add_unk': self._add_unk,
                'unk_token': self._unk_token}

    @classmethod
    def from_serializable(cls, contents):
        """ 직렬화된 딕셔너리에서 Vocabulary 객체를 만듭니다 """
        return cls(**contents)

    def add_token(self, token):
        """ 토큰을 기반으로 매핑 딕셔너리를 업데이트합니다

        매개변수:
            token (str): Vocabulary에 추가할 토큰
        반환값:
            index (int): 토큰에 상응하는 정수
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        """ 토큰 리스트를 Vocabulary에 추가합니다.

        매개변수:
            tokens (list): 문자열 토큰 리스트
        반환값:
            indices (list): 토큰 리스트에 상응되는 인덱스 리스트
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """ 토큰에 대응하는 인덱스를 추출합니다.
        토큰이 없으면 UNK 인덱스를 반환합니다.

        매개변수:
            token (str): 찾을 토큰
        반환값:
            index (int): 토큰에 해당하는 인덱스
        노트:
            UNK 토큰을 사용하려면 (Vocabulary에 추가하기 위해)
            `unk_index`가 0보다 커야 합니다.
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """ 인덱스에 해당하는 토큰을 반환합니다.

        매개변수:
            index (int): 찾을 인덱스
        반환값:
            token (str): 인텍스에 해당하는 토큰
        에러:
            KeyError: 인덱스가 Vocabulary에 없을 때 발생합니다.
        """
        if index not in self._idx_to_token:
            raise KeyError("Vocabulary에 인덱스(%d)가 없습니다." % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


class ReviewVectorizer(object):
    """ 어휘 사전을 생성하고 관리합니다 """

    def __init__(self, review_vocab, rating_vocab):
        """
        매개변수:
            review_vocab (Vocabulary): 단어를 정수에 매핑하는 Vocabulary
            rating_vocab (Vocabulary): 클래스 레이블을 정수에 매핑하는 Vocabulary
        """
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    def vectorize(self, review):
        """ 리뷰에 대한 웟-핫 벡터를 만듭니다

        매개변수:
            review (str): 리뷰
        반환값:
            one_hot (np.ndarray): 원-핫 벡터
        """
        one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)

        for token in review.split(" "):
            if token not in string.punctuation:
                one_hot[self.review_vocab.lookup_token(token)] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        """ 데이터셋 데이터프레임에서 Vectorizer 객체를 만듭니다

        매개변수:
            review_df (pandas.DataFrame): 리뷰 데이터셋
            cutoff (int): 빈도 기반 필터링 설정값
        반환값:
            ReviewVectorizer 객체
        """
        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)

        # 점수를 추가합니다
        for rating in sorted(set(review_df.rating)):
            rating_vocab.add_token(rating)

        # count > cutoff인 단어를 추가합니다
        word_counts = Counter()
        for review in review_df.review:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)

        return cls(review_vocab, rating_vocab)

    @classmethod
    def from_serializable(cls, contents):
        """ 직렬화된 딕셔너리에서 ReviewVectorizer 객체를 만듭니다

        매개변수:
            contents (dict): 직렬화된 딕셔너리
        반환값:
            ReviewVectorizer 클래스 객체
        """
        review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
        rating_vocab =  Vocabulary.from_serializable(contents['rating_vocab'])

        return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)

    def to_serializable(self):
        """ 캐싱을 위해 직렬화된 딕셔너리를 만듭니다

        반환값:
            contents (dict): 직렬화된 딕셔너리
        """
        return {'review_vocab': self.review_vocab.to_serializable(),
                'rating_vocab': self.rating_vocab.to_serializable()}


class ReviewClassifier(nn.Module):
    def __init__(self, num_features, num_hidden_features):
        super(ReviewClassifier, self).__init__()

        fc1_out = num_hidden_features if num_hidden_features > 0 else 1
        self.fc1 = nn.Linear(in_features=num_features, out_features=fc1_out)
        if num_hidden_features > 0:
            self.activation1 = nn.ReLU()
            self.dropout2 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(in_features=num_hidden_features, out_features=1)
        else :
            self.fc2 = None
        self.activation2 = nn.Sigmoid()


    def forward(self, x_in, apply_sigmoid=False):
        y_out = self.fc1(x_in)
        if self.fc2 is not None:
            y_out = self.activation1(y_out)
            y_out = self.dropout2(y_out)
            y_out = self.fc2(y_out)
        y_out = y_out.squeeze()
        if apply_sigmoid:
            y_out = self.activation2(y_out)
        return y_out


if __name__ == '__main__':
    def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
        """
        파이토치 DataLoader를 감싸고 있는 제너레이터 함수.
        걱 텐서를 지정된 장치로 이동합니다.
        """
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                shuffle=shuffle, drop_last=drop_last)

        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict

    def compute_accuracy(y_pred, y_target):
        y_target = y_target.cpu()
        y_pred_indices = (torch.sigmoid(y_pred)>0.5).cpu().long()#.max(dim=1)[1]
        n_correct = torch.eq(y_pred_indices, y_target).sum().item()
        return n_correct / len(y_pred_indices) * 100

    def make_train_state(args):
        return {'stop_early': False,
                'early_stopping_step': 0,
                'early_stopping_best_val': 1e8,
                'learning_rate': args.learning_rate,
                'epoch_index': 0,
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'test_loss': -1,
                'test_acc': -1,
                'model_filename': args.model_state_file}
    train_state = make_train_state(args)

    def epoch(epoch_type) -> tuple:
        dataset.set_split(epoch_type)
        batch_generator = generate_batches(dataset, args.batch_size, device=args.device)
        running_loss = 0.0
        running_acc = 0.0

        for batch_index, batch_dict in enumerate(batch_generator):
            if epoch_type == 'train':
                optimizer.zero_grad()

            y_pred = classifier(batch_dict['x_data'].float())
            loss = lf(y_pred, batch_dict['y_target'].float())
            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            if epoch_type == 'train':
                loss.backward()
                optimizer.step()

            acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_batch - running_acc) / (batch_index + 1)


        if epoch_type != 'test':
            train_state[f'{epoch_type}_loss'].append(running_loss)
            train_state[f'{epoch_type}_acc'].append(running_acc)

        return running_loss, running_acc

    args.device = torch.device(args.device)

    if not args.reload_from_files:
        dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
        vectorizer = dataset.get_vectorizer()

        classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab), num_hidden_features=args.num_hidden_features)
        classifier = classifier.to(args.device)

        lf = nn.BCEWithLogitsLoss().to(args.device)
        optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

        start = time.time()
        for epoch_index in range(args.num_epochs):
            train_state['epoch_index'] = epoch_index

            classifier.train()
            epoch('train')

            classifier.eval()
            loss, acc = epoch('val')

            print(f'validation : {loss:.3f} / {acc:.2f}')

        fin = time.time()
        print(f'elapsed    : {fin - start:.2f} s')
        classifier.eval()
        loss, acc = epoch('test')
        train_state['test_loss'], train_state['train_acc'] = loss, acc
        print(f'test       : {loss:.3f} / {acc:.2f}')

        print(train_state)
        torch.save(classifier.state_dict(), args.save_dir + args.model_state_file)
        with open(args.save_dir + args.vectorizer_file, "w") as f:
            json.dump(vectorizer.to_serializable(), f)

    with open(args.save_dir + args.vectorizer_file) as f:
        vectorizer = ReviewVectorizer.from_serializable(json.load(f))

    classifier = ReviewClassifier(len(vectorizer.review_vocab), args.num_hidden_features)
    classifier.load_state_dict(torch.load(args.save_dir + args.model_state_file))
    classifier = classifier.to(args.device)
    classifier.eval()

    reviews = [
        "This is a pretty awesome book",
        "Great Great Great Fucking bad",
        "Fucking good",
        "holly shit"
    ]
    for review in reviews:
        review_vector = torch.tensor(vectorizer.vectorize(review)).to(args.device)
        y_pred = (classifier(review_vector, apply_sigmoid=True) > 0.5).long().item() # (torch.sigmoid(classifier(review_vector)) > 0.5).long().item()
        review_rate = vectorizer.rating_vocab.lookup_index(y_pred)
        print(f'{review} -> {review_rate}')
