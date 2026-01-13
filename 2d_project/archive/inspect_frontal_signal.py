import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('/data/gait/data/1/1-1_front_pose_fps23.csv')
    
    # Plot potential periodic signals
    plt.figure(figsize=(10, 10))
    
    # 1. Heel Y (Vertical - usually best for sagittal, maybe frontal too)
    plt.subplot(4,1,1)
    plt.plot(df['RIGHT_HEEL_y'][:200])
    plt.title('Heel Y (Vertical)')
    
    # 2. Ankle X (Lateral - might show step width variation)
    plt.subplot(4,1,2)
    plt.plot(df['RIGHT_ANKLE_x'][:200])
    plt.title('Ankle X (Lateral)')
    
    # 3. Knee X (Varus/Valgus sway)
    plt.subplot(4,1,3)
    plt.plot(df['RIGHT_KNEE_x'][:200])
    plt.title('Knee X (Lateral)')

    # 4. Hip Y (Vertical CoM)
    plt.subplot(4,1,4)
    plt.plot(df['RIGHT_HIP_y'][:200])
    plt.title('Hip Y (Vertical)')
    
    plt.tight_layout()
    plt.savefig('frontal_signal_inspection.png')
    print("Saved frontal_signal_inspection.png")
    
except Exception as e:
    print(e)
