import json
import os


def main():
    test_set = json.load(open(os.path.join('./Multi_Teacher_In_Questions/preprocessed_data/', 'count_test.json')))
    res = []
    for i in range(len(test_set)):
        answer = test_set[i]['answer']
        res.append(answer)
    print(len(res))

    pred_fn = './Multi_Teacher_In_Questions/circle_count_encoder_multihop_guide_count_again/output/predict_result_8220/predict.txt'
    pred = [x.strip() for x in open(pred_fn).readlines()] # one prediction per line

    print(len(pred))

  
    count = 0
    for i in range(len(test_set)):
        if res[i] == pred[i]:
            count += 1

    print(count / len(pred))


if __name__ == '__main__':
    main()
