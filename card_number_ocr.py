#!/usr/bin/env python3
"""
宝可梦卡牌编号识别脚本
使用YOLO-like目标检测 + CRNN文字识别
"""

import os
import cv2
import numpy as np
import easyocr
from pathlib import Path
from typing import List, Tuple, Optional


class CardNumberDetector:
    """基于OpenCV的卡片编号区域检测器 (YOLO-like)"""
    
    def __init__(self):
        # 定义左下角编号区域的大致位置（相对坐标）
        # 基于宝可梦卡牌的固定布局 - 编号在左下角，格式如 "086/080 AR"
        self.roi_x_ratio = (0.15, 0.60)  # 左下角水平范围（包含AR后缀）
        self.roi_y_ratio = (0.935, 0.985)  # 底部垂直范围（编号所在位置）
        
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测卡片编号区域
        返回: [(x1, y1, x2, y2), ...]
        """
        h, w = image.shape[:2]
        
        # 基于先验知识，直接定位左下角区域
        x1 = int(w * self.roi_x_ratio[0])
        y1 = int(h * self.roi_y_ratio[0])
        x2 = int(w * self.roi_x_ratio[1])
        y2 = int(h * self.roi_y_ratio[1])
        
        # 裁剪ROI区域
        roi = image[y1:y2, x1:x2]
        
        # 使用颜色分割和轮廓检测来精确定位编号区域
        refined_boxes = self._refine_detection(roi, x1, y1)
        
        return refined_boxes if refined_boxes else [(x1, y1, x2, y2)]
    
    def _refine_detection(self, roi: np.ndarray, offset_x: int, offset_y: int) -> List[Tuple[int, int, int, int]]:
        """使用图像处理精确定位文字区域"""
        # 转为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 二值化 - 白色文字在深色背景上
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # 形态学操作连接文字区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 过滤小区域和不符合比例的区域
            if w > 50 and h > 10 and w / h > 2:
                boxes.append((x + offset_x, y + offset_y, x + w + offset_x, y + h + offset_y))
        
        return boxes


class CRNNRecognizer:
    """基于EasyOCR的CRNN文字识别器"""
    
    def __init__(self, languages: List[str] = None):
        """
        初始化CRNN识别器
        Args:
            languages: 语言列表，默认 ['en'] 用于识别数字和字母
        """
        if languages is None:
            languages = ['en']
        self.reader = easyocr.Reader(languages, gpu=False, verbose=False)
        
    def recognize(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """
        识别图像中的文字
        返回: [(text, confidence), ...]
        """
        results = self.reader.readtext(image)
        return [(text, conf) for (_, text, conf) in results]


class CardNumberOCR:
    """完整的卡片编号识别流程"""
    
    def __init__(self):
        self.detector = CardNumberDetector()
        self.recognizer = CRNNRecognizer(languages=['en'])
        
    def process(self, image_path: str, visualize: bool = False) -> Optional[str]:
        """
        处理单张图片
        Args:
            image_path: 图片路径
            visualize: 是否可视化结果
        Returns:
            识别到的卡片编号字符串
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图片 {image_path}")
            return None
        
        # 1. 检测编号区域 (YOLO-like)
        boxes = self.detector.detect(image)
        
        all_recognized_texts = []  # 收集所有识别到的原始文本
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # 裁剪区域
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # 2. CRNN文字识别（使用多种预处理方法）
            texts = self._recognize_with_enhancement(crop)
            all_recognized_texts.extend(texts)
            
            # 可视化
            if visualize:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if texts:
                    cv2.putText(image, texts[0][0], (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if visualize:
            output_path = image_path.rsplit('.', 1)[0] + '_result.jpg'
            cv2.imwrite(output_path, image)
            print(f"可视化结果已保存: {output_path}")
        
        # 综合所有识别结果，找到最佳编号和后缀
        if all_recognized_texts:
            return self._combine_results(all_recognized_texts)
        
        # 备用方案：直接识别整个左下角区域
        return self._fallback_recognize(image)
    
    def _combine_results(self, all_texts: List[Tuple[str, float]]) -> Optional[str]:
        """综合多个识别结果，提取最佳编号和可能的后缀"""
        # 找到置信度最高的数字编号
        best_number = None
        best_conf = 0
        
        for text, conf in all_texts:
            formatted = self._format_card_number(text)
            if formatted and conf > best_conf:
                best_number = formatted
                best_conf = conf
        
        if not best_number:
            return all_texts[0][0] if all_texts else None
        
        # 如果最佳编号已经包含后缀，直接返回
        import re
        if re.search(r'\d{2,3}/\d{2,3}[A-Z]{2,}$', best_number):
            return best_number
        
        # 尝试从所有结果中提取后缀
        suffix = ""
        for text, _ in all_texts:
            potential_suffix = self._extract_suffix(text)
            if potential_suffix and len(potential_suffix) >= 2:
                suffix = potential_suffix
                break
        
        # 合并编号和后缀
        if suffix and not best_number.endswith(suffix):
            return best_number + suffix
        
        return best_number
    
    def _format_card_number(self, text: str) -> Optional[str]:
        """格式化卡片编号"""
        import re
        # 清理文字
        text = text.strip().replace(' ', '').replace('O', '0')
        # 匹配模式: 数字/数字 或 数字/数字+字母 (支持AR, SAR等后缀)
        pattern = r'\d{2,3}/\d{2,3}(?:[A-Z]+)?'
        match = re.search(pattern, text)
        if match:
            return match.group()
        # 尝试更宽松的匹配
        pattern2 = r'\d{2,3}[/.]\d{2,3}'
        match2 = re.search(pattern2, text)
        if match2:
            result = match2.group().replace('.', '/')
            # 尝试找到后面的字母后缀
            rest = text[match2.end():].strip()
            alpha_suffix = ''
            for c in rest:
                if c.isalpha():
                    alpha_suffix += c.upper()
                elif alpha_suffix:
                    break
            return result + alpha_suffix
        return None
    
    def _recognize_with_enhancement(self, crop: np.ndarray) -> List[Tuple[str, float]]:
        """使用多种预处理方法进行识别，返回所有结果"""
        all_results = []
        
        # 1. 原图识别
        results = self.recognizer.recognize(crop)
        all_results.extend(results)
        
        # 2. 灰度图识别（通常效果最好）
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        results = self.recognizer.recognize(gray)
        all_results.extend(results)
        
        # 3. 放大后识别（有助于识别小字如AR）
        h, w = crop.shape[:2]
        resized = cv2.resize(crop, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
        results = self.recognizer.recognize(resized)
        all_results.extend(results)
        
        # 4. 灰度+放大
        gray_resized = cv2.resize(gray, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
        results = self.recognizer.recognize(gray_resized)
        all_results.extend(results)
        
        return all_results
    
    def _extract_suffix(self, text: str) -> str:
        """从识别结果中提取字母后缀（如AR, SAR等）"""
        import re
        
        # 常见后缀白名单（按优先级排序）
        valid_suffixes = ['SAR', 'AR', 'SR', 'HR', 'UR', 'RR', 'CHR']
        text_upper = text.upper()
        
        # 首先直接检查白名单（在原始文本中）
        for suffix in valid_suffixes:
            if suffix in text_upper:
                return suffix
        
        # 查找数字/数字后跟着的字母序列（必须包含至少2个字母）
        # 模式：数字/数字后跟着字母（处理086/O804R这样的情况）
        match = re.search(r'\d{2,3}[/.]\d+([A-Za-z]{2,})', text)
        if match:
            raw_suffix = match.group(1)
            # 清理误识别：0->O, 4->A, 8->B等
            cleaned = raw_suffix.upper()
            cleaned = cleaned.replace('0', 'O')  # 0通常是O的误识别
            cleaned = cleaned.replace('4', 'A')  # 4可能是A的误识别
            cleaned = cleaned.replace('1', 'I')  # 1可能是I的误识别
            cleaned = cleaned.replace('5', 'S')  # 5可能是S的误识别
            
            # 再次检查白名单
            for suffix in valid_suffixes:
                if suffix in cleaned:
                    return suffix
            
            # 提取纯字母
            letters_only = re.sub(r'[^A-Z]', '', cleaned)
            if len(letters_only) >= 2:
                return letters_only[-3:] if len(letters_only) > 3 else letters_only
        
        # 处理特殊情况：086/O804R -> 从数字混合中提取AR
        mixed_match = re.search(r'\d{2,3}[/.][A-Za-z0-9]+', text)
        if mixed_match:
            mixed = mixed_match.group(0)
            # 提取所有字母
            letters = re.findall(r'[A-Z]', mixed.upper())
            if len(letters) >= 2:
                potential = ''.join(letters)
                # 检查是否是有效后缀的变体
                if 'AR' in potential:
                    return 'AR'
                if potential == 'OR' or potential == 'QR':  # OR/QR可能是AR的误识别
                    return 'AR'
        
        return ""
    
    def _fallback_recognize(self, image: np.ndarray) -> Optional[str]:
        """备用识别方案 - 直接裁剪编号区域"""
        h, w = image.shape[:2]
        # 精确裁剪编号区域 (基于宝可梦卡牌布局)
        x1 = int(w * 0.15)
        y1 = int(h * 0.935)
        x2 = int(w * 0.60)  # 扩展以包含AR后缀
        y2 = int(h * 0.985)
        
        crop = image[y1:y2, x1:x2]
        
        # 使用多种预处理方法识别
        all_texts = self._recognize_with_enhancement(crop)
        
        # 首先找到数字编号（通常是置信度最高的）
        best_number = None
        best_conf = 0
        all_raw_texts = []
        
        for text, conf in all_texts:
            all_raw_texts.append(text)
            formatted = self._format_card_number(text)
            if formatted and conf > best_conf:
                best_number = formatted
                best_conf = conf
        
        # 如果没有找到数字编号，返回原始结果
        if not best_number:
            return all_texts[0][0] if all_texts else None
        
        # 尝试从所有识别结果中提取后缀
        suffix = ""
        for text, _ in all_texts:
            potential_suffix = self._extract_suffix(text)
            if potential_suffix:
                suffix = potential_suffix
                break
        
        # 检查后缀是否已经在编号中
        if suffix and not best_number.endswith(suffix):
            best_number += suffix
        
        return best_number


def process_directory(input_dir: str, visualize: bool = False):
    """
    处理目录中的所有图片
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 目录不存在 {input_dir}")
        return
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"在 {input_dir} 中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片，开始处理...\n")
    
    ocr = CardNumberOCR()
    
    for img_path in sorted(image_files):
        print(f"处理: {img_path.name}")
        result = ocr.process(str(img_path), visualize=visualize)
        if result:
            print(f"  -> 卡片编号: {result}")
        else:
            print(f"  -> 未能识别编号")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='宝可梦卡牌编号识别 (YOLO+CRNN)')
    parser.add_argument('--input', '-i', default='images',
                       help='输入图片目录 (默认: images)')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='保存可视化结果')
    
    args = parser.parse_args()
    
    process_directory(args.input, visualize=args.visualize)
