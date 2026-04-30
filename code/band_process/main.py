import argparse
import os.path
import time
import traceback

import cv2
import numpy as np
import tqdm
from cv2_rolling_ball import subtract_background_rolling_ball
from loguru import logger
from ultralytics import YOLO

import utils
from rag_api_utils import get_code, chat

RESIZE_LENGTH = 2200
BASELINE_RESIZE_LENGTH = 1850
CONTROL_HIV_2_THRESHOLD = 1400
color_dict = {
    0: [0, 0, 255],
    1: [0, 255, 0]

}


def main(image_path=None, show=True):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    box_model_path = os.path.join(project_root, "box_best.pt")
    box_model = YOLO(box_model_path, task='detect')
    source_root_path = os.path.join(project_root, "dataset", "sample", "picture")
    target_path = os.path.join(project_root, "dataset", "dataset_2024-12-21")
    if image_path:
        if os.path.isabs(image_path):
            image_file_list = [image_path]
        else:
            image_file_list = [os.path.join(source_root_path, image_path)]
    else:
        image_file_list = [os.path.join(source_root_path, name) for name in sorted(os.listdir(source_root_path))]
    for source_file_path in tqdm.tqdm(image_file_list):
        image_file_name = os.path.basename(source_file_path)
        try:
            # 读取图片
            # original_image = cv2.imread(source_file_path)
            original_image = cv2.imdecode(np.fromfile(source_file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            # cv2.imshow("0", cv2.resize(original_image, (1920, 240)))
            # cv2.waitKey(0)
            # continue
            # 将图片灰度化
            original_gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            # cv2.imshow("1", cv2.resize(original_gray_image, (1920, 240)))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # continue

            # 裁剪、矫正
            image = utils.crop_rotate_image(original_gray_image)
            
            if image is None:
                continue
            # new_image_path = os.path.join(target_path, image_file_name)
            # cv2.imwrite(new_image_path, image)
            # cv2.imshow("original_image", cv2.resize(original_image, (1920, 240)))
            # cv2.imshow("processed_image", cv2.resize(image, (1095, 75)))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # continue
            image = cv2.resize(image, (RESIZE_LENGTH, image.shape[0]), interpolation=cv2.INTER_AREA)
            _image = image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image_vis = image.copy()
            result = box_model(image.copy(), imgsz=1280, verbose=False)[0].boxes.cpu().numpy().data
            print(result)
            # cv2.imshow("processed_image", cv2.resize(image, (1095, 75)))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # exit()

            for box in result:
                if box[4] < 0.3:
                    continue
                x1, y1, x2, y2 = box[:4].astype(int)
                image_vis = cv2.rectangle(image_vis, (x1, y1), (x2, y2), color_dict.get(int(box[5])), 2)
            if show:
                cv2.imshow("2", cv2.resize(image_vis, (1095, 75)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # continue

            digital_object_detection_result, black_object_detection_result = utils.get_object_detection_result(result)
            # print(digital_object_detection_result, black_object_detection_result)
            digital_flag, digital_position_result = utils.recognize_digital(digital_object_detection_result,
                                                                            black_object_detection_result,
                                                                            image.shape[1])
            # print(digital_flag, digital_position_result)
            black_position_result, image = utils.get_position_result(digital_flag, black_object_detection_result, image)
            # print(black_position_result)
            image = image.copy()
            # for box in black_position_result:
            #     if box[2] < 0.3:
            #         continue
            #     x1, x2 = box[:2]
            #     image = cv2.rectangle(image, (int(x1), 0), (int(x2), image.shape[0]), color_dict.get(1), 2)
            #
            # cv2.imshow("3", cv2.resize(image, (1095, 75)))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(digital_flag, black_position_result)

            baseline_index = utils.get_imt_baseline_begin_position(digital_position_result,
                                                                   black_position_result)
            # print(image_file_name, baseline_index)

            # image_baseline_left = image[:, : baseline_index]
            image_baseline_right = image[:, baseline_index:]
            rate = BASELINE_RESIZE_LENGTH / image_baseline_right.shape[1]
            # image_baseline_right = cv2.resize(image_baseline_right,
            #                                   (BASELINE_RESIZE_LENGTH, image_baseline_right.shape[0]),
            #                                   interpolation=cv2.INTER_AREA)
            black_position_result_ = black_position_result.copy()
            black_position_result = []
            for i in black_position_result_:
                black_position_result.append(((i[0] - baseline_index) * rate + baseline_index,
                                              (i[1] - baseline_index) * rate + baseline_index, i[2]))
            # image = np.concatenate((image_baseline_left, image_baseline_right), axis=1)

            image_subtract, background = subtract_background_rolling_ball(_image, 150, light_background=True,
                                                                          use_paraboloid=False, do_presmooth=True)
            image_subtract_normalize = image_subtract.copy()
            cv2.normalize(image_subtract, image_subtract_normalize, 0, 255, cv2.NORM_MINMAX)
            image_subtract_normalize_average_gray = utils.get_y_average_gray(image_subtract_normalize)

            tip, judge_point_distance = utils.get_imt_tip_and_distance(digital_position_result, black_position_result,
                                                                       image_subtract_normalize_average_gray)
            # print(tip, judge_point_distance)
            # print(image_file_name, judge_point_distance)
            # continue
            mark_point_list = []
            for point_symbol, point in zip(tip, judge_point_distance):
                if point_symbol == '+':
                    mark_point_list.append(point)

            key_word = fr"根据知识库描述：一个病人的艾滋病检测条带的结果如下：该检测条带上包含{'、'.join(mark_point_list)}等点位，那么这个病人的检测条带属于什么类型"
            print(f"{image_file_name} 的 标记点为 {key_word}")
            # continue
            message = [
                {
                    "dataId": get_code(32),
                    "role": "user",
                    "content": key_word
                }
            ]
            current_time = time.time()
            c_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
            chat_id = get_code(8)
            share_id = "1gt3ytp7xvbjb1m56sw6wn5q"
            result = chat(message, c_time, chat_id, share_id)
            logger.info(f"{result['choices'][0]['message']['content']}")


        except Exception as e:
            logger.error(f"矫正错误{e} {source_file_path}")
            traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="单图文件名或绝对路径")
    parser.add_argument("--no-show", action="store_true", help="不显示可视化窗口")
    args = parser.parse_args()
    main(image_path=args.image, show=not args.no_show)
