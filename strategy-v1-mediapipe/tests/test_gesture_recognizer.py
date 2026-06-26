"""GestureRecognizer 单元测试"""

import sys
sys.path.insert(0, '..')

from airmouse.gesture_recognizer import Gesture, classify_gesture, get_fingertip_position

# 模拟 MediaPipe landmark 对象
class FakeLM:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

def make_flat_hand():
    """模拟一张张开的手：指尖 y 小于指根 y（在屏幕上更高）"""
    return [
        FakeLM(0.5, 0.8, 0),   # 0: wrist
        FakeLM(0.45, 0.7, 0), FakeLM(0.43, 0.6, 0), FakeLM(0.41, 0.5, 0), FakeLM(0.39, 0.4, 0),  # thumb
        FakeLM(0.5, 0.7, 0),   # 5: index MCP
        FakeLM(0.51, 0.55, 0), FakeLM(0.52, 0.45, 0), FakeLM(0.53, 0.35, 0),  # index
        FakeLM(0.55, 0.7, 0),  # 9: middle MCP
        FakeLM(0.56, 0.55, 0), FakeLM(0.57, 0.45, 0), FakeLM(0.58, 0.35, 0),  # middle
        FakeLM(0.6, 0.7, 0),   # 13: ring MCP
        FakeLM(0.61, 0.55, 0), FakeLM(0.62, 0.45, 0), FakeLM(0.63, 0.35, 0),  # ring
        FakeLM(0.65, 0.7, 0),  # 17: pinky MCP
        FakeLM(0.66, 0.6, 0), FakeLM(0.67, 0.5, 0), FakeLM(0.68, 0.4, 0),   # pinky
    ]

def make_pointing_hand():
    """模拟食指指向：食指伸直，其他手指弯曲"""
    lms = []
    lms.append(FakeLM(0.5, 0.8, 0))    # 0: wrist
    # 拇指（弯曲靠近手掌）
    lms.append(FakeLM(0.48, 0.72, 0))
    lms.append(FakeLM(0.47, 0.74, 0))
    lms.append(FakeLM(0.46, 0.76, 0))
    lms.append(FakeLM(0.45, 0.75, 0))  # 4: thumb tip (靠近手掌)
    # 食指（伸直向上）
    lms.append(FakeLM(0.5, 0.65, 0))   # 5: index MCP
    lms.append(FakeLM(0.5, 0.5, 0))
    lms.append(FakeLM(0.5, 0.3, 0))
    lms.append(FakeLM(0.5, 0.1, 0))    # 8: index tip (高高在上)
    # 中指（弯曲，尖端靠近 MCP）
    lms.append(FakeLM(0.55, 0.65, 0))  # 9: middle MCP
    lms.append(FakeLM(0.58, 0.63, 0))
    lms.append(FakeLM(0.6, 0.62, 0))
    lms.append(FakeLM(0.62, 0.63, 0))  # 12: middle tip (靠近 MCP，弯曲)
    # 无名指（弯曲）
    lms.append(FakeLM(0.6, 0.65, 0))   # 13: ring MCP
    lms.append(FakeLM(0.63, 0.63, 0))
    lms.append(FakeLM(0.66, 0.62, 0))
    lms.append(FakeLM(0.69, 0.63, 0))  # 16: ring tip (靠近 MCP)
    # 小指（弯曲）
    lms.append(FakeLM(0.65, 0.65, 0))  # 17: pinky MCP
    lms.append(FakeLM(0.68, 0.63, 0))
    lms.append(FakeLM(0.71, 0.62, 0))
    lms.append(FakeLM(0.74, 0.63, 0))  # 20: pinky tip (靠近 MCP)
    return lms

def make_pinch_thumb_index():
    """模拟拇指+食指捏合"""
    lms = make_pointing_hand()
    # 把拇指尖移到食指尖旁边
    lms[4] = FakeLM(0.49, 0.11, 0)  # thumb tip 紧挨 index tip (0.5, 0.1)
    return lms

def make_pinch_thumb_middle():
    """模拟拇指+中指捏合"""
    lms = make_pointing_hand()
    # 拇指尖靠近中指尖(0.62, 0.63)
    lms[4] = FakeLM(0.61, 0.62, 0)
    return lms

def make_fist():
    """模拟握拳：所有指尖靠近手腕"""
    WRIST = FakeLM(0.5, 0.8, 0)
    return [
        WRIST,                                                        # 0
        FakeLM(0.49, 0.78, 0), FakeLM(0.48, 0.79, 0), FakeLM(0.47, 0.8, 0), FakeLM(0.46, 0.79, 0),  # thumb
        FakeLM(0.5, 0.78, 0),                                         # 5: index MCP
        FakeLM(0.48, 0.79, 0), FakeLM(0.47, 0.8, 0), FakeLM(0.46, 0.79, 0),  # index
        FakeLM(0.52, 0.78, 0),                                        # 9: middle MCP
        FakeLM(0.5, 0.79, 0), FakeLM(0.49, 0.8, 0), FakeLM(0.48, 0.79, 0),  # middle
        FakeLM(0.54, 0.78, 0),                                        # 13: ring MCP
        FakeLM(0.52, 0.79, 0), FakeLM(0.51, 0.8, 0), FakeLM(0.5, 0.79, 0),   # ring
        FakeLM(0.56, 0.78, 0),                                        # 17: pinky MCP
        FakeLM(0.54, 0.79, 0), FakeLM(0.53, 0.8, 0), FakeLM(0.52, 0.79, 0),  # pinky
    ]


def test_pointing():
    result = classify_gesture(make_pointing_hand())
    assert result == Gesture.POINTING, f"Expected POINTING, got {result}"

def test_pinch_index():
    result = classify_gesture(make_pinch_thumb_index())
    assert result == Gesture.PINCH_THUMB_INDEX, f"Expected PINCH_THUMB_INDEX, got {result}"

def test_pinch_middle():
    result = classify_gesture(make_pinch_thumb_middle())
    assert result == Gesture.PINCH_THUMB_MIDDLE, f"Expected PINCH_THUMB_MIDDLE, got {result}"

def test_fist():
    result = classify_gesture(make_fist())
    assert result == Gesture.FIST, f"Expected FIST, got {result}"

def test_flat_hand():
    """全张开的手不应被识别为特定手势（既非捏合也非握拳也非指向）"""
    result = classify_gesture(make_flat_hand())
    # 全张开可能是 NONE 或 POINTING（取决于食指是否明显高于指根）
    assert result not in [Gesture.PINCH_THUMB_INDEX, Gesture.PINCH_THUMB_MIDDLE, Gesture.FIST], \
        f"Flat hand should not be pinch or fist, got {result}"


def test_get_fingertip_position():
    from airmouse.gesture_recognizer import get_fingertip_position
    lms = make_pointing_hand()
    pos = get_fingertip_position(lms)
    assert pos is not None
    assert abs(pos[0] - 0.5) < 0.01  # index tip x from pointing hand
    assert abs(pos[1] - 0.1) < 0.01  # index tip y from pointing hand
    # None case
    assert get_fingertip_position(None) is None
    assert get_fingertip_position([]) is None
    print("✓ test_get_fingertip_position passed")


if __name__ == '__main__':
    test_pointing()
    print("✓ test_pointing passed")
    test_pinch_index()
    print("✓ test_pinch_index passed")
    test_pinch_middle()
    print("✓ test_pinch_middle passed")
    test_fist()
    print("✓ test_fist passed")
    test_flat_hand()
    print("✓ test_flat_hand passed")
    test_get_fingertip_position()
    print("\n所有测试通过!")
