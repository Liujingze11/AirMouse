"""GestureStateMachine 单元测试"""

import sys
sys.path.insert(0, '..')

from airmouse.gesture_recognizer import (
    Gesture, AppState, GestureStateMachine,
    FIST_HOLD_FRAMES, PINCH_HOLD_FRAMES,
)


def test_initial_state():
    sm = GestureStateMachine()
    assert sm.state == AppState.PAUSED, "Initial state should be PAUSED"


def test_fist_activates():
    sm = GestureStateMachine()
    # 发送 FIST_HOLD_FRAMES 帧握拳
    for _ in range(FIST_HOLD_FRAMES):
        state = sm.update(Gesture.FIST)
    assert state == AppState.ACTIVE, f"Should be ACTIVE after holding fist, got {state}"


def test_fist_pauses():
    sm = GestureStateMachine()
    # 先激活
    for _ in range(FIST_HOLD_FRAMES):
        sm.update(Gesture.FIST)
    assert sm.state == AppState.ACTIVE
    # 再握拳暂停
    for _ in range(FIST_HOLD_FRAMES):
        state = sm.update(Gesture.FIST)
    assert state == AppState.PAUSED, f"Should be PAUSED, got {state}"


def test_fist_interrupted():
    """握拳不够久不会触发"""
    sm = GestureStateMachine()
    for _ in range(FIST_HOLD_FRAMES - 1):
        state = sm.update(Gesture.FIST)
    assert state == AppState.PAUSED, "Should stay PAUSED if fist not held long enough"


def test_pinch_left_down():
    sm = GestureStateMachine()
    # 激活
    for _ in range(FIST_HOLD_FRAMES):
        sm.update(Gesture.FIST)
    # 捏合防抖
    for _ in range(PINCH_HOLD_FRAMES):
        state = sm.update(Gesture.PINCH_THUMB_INDEX)
    assert state == AppState.LEFT_DOWN, f"Should be LEFT_DOWN, got {state}"


def test_pinch_release():
    sm = GestureStateMachine()
    for _ in range(FIST_HOLD_FRAMES):
        sm.update(Gesture.FIST)
    for _ in range(PINCH_HOLD_FRAMES):
        sm.update(Gesture.PINCH_THUMB_INDEX)
    assert sm.state == AppState.LEFT_DOWN
    # 松开捏合
    state = sm.update(Gesture.POINTING)
    assert state == AppState.ACTIVE, f"Should return to ACTIVE on release, got {state}"


def test_dragging():
    sm = GestureStateMachine()
    for _ in range(FIST_HOLD_FRAMES):
        sm.update(Gesture.FIST)
    for _ in range(PINCH_HOLD_FRAMES):
        sm.update(Gesture.PINCH_THUMB_INDEX)
    assert sm.state == AppState.LEFT_DOWN
    # 继续保持捏合
    state = sm.update(Gesture.PINCH_THUMB_INDEX)
    assert state == AppState.DRAGGING, f"Should be DRAGGING, got {state}"


def test_ignores_gestures_when_paused():
    sm = GestureStateMachine()
    # 暂停状态下，捏合不应该有反应
    for _ in range(PINCH_HOLD_FRAMES):
        state = sm.update(Gesture.PINCH_THUMB_INDEX)
    assert state == AppState.PAUSED, f"Should stay PAUSED, got {state}"


if __name__ == '__main__':
    test_initial_state()
    print("✓ test_initial_state passed")
    test_fist_activates()
    print("✓ test_fist_activates passed")
    test_fist_pauses()
    print("✓ test_fist_pauses passed")
    test_fist_interrupted()
    print("✓ test_fist_interrupted passed")
    test_pinch_left_down()
    print("✓ test_pinch_left_down passed")
    test_pinch_release()
    print("✓ test_pinch_release passed")
    test_dragging()
    print("✓ test_dragging passed")
    test_ignores_gestures_when_paused()
    print("✓ test_ignores_gestures_when_paused passed")
    print("\n所有状态机测试通过!")
