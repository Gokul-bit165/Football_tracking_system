import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanBallFilter:
    """
    Kalman filter tracking and interpolation for the football using a 2D constant velocity model.
    """
    def __init__(self, dt: float = 0.04, process_noise: float = 0.1, measurement_noise: float = 2.0):
        """
        Initialize the state transition and covariance matrices.
        
        Args:
            dt (float): Time interval between frames (default 0.04s for 25 FPS).
            process_noise (float): Process noise multiplier (controls velocity variance).
            measurement_noise (float): Measurement noise standard deviation (in pixels).
        """
        self.dt = dt
        
        # 4D state vector: [x, y, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix F
        self.kf.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Measurement matrix H
        self.kf.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        # Measurement noise covariance matrix R
        self.kf.R = np.eye(2, dtype=np.float32) * (measurement_noise ** 2)
        
        # Process noise covariance matrix Q
        # Using discrete constant white noise approximation for 2D position and velocity
        q_pos = (dt**4)/4.0 * process_noise
        q_vel = (dt**2) * process_noise
        q_pos_vel = (dt**3)/2.0 * process_noise
        
        self.kf.Q = np.array([
            [q_pos,  0.0,   q_pos_vel, 0.0],
            [0.0,   q_pos,  0.0,       q_pos_vel],
            [q_pos_vel, 0.0,   q_vel,     0.0],
            [0.0,   q_pos_vel, 0.0,       q_vel]
        ], dtype=np.float32)
        
        # Initial state covariance P
        self.kf.P = np.eye(4, dtype=np.float32) * 500.0
        
        # Initial state x: centered at 0 with 0 velocity
        self.kf.x = np.zeros(4, dtype=np.float32)
        
        self.initialized = False

    def initialize(self, x: float, y: float):
        """
        Set initial state position.
        
        Args:
            x (float): Initial x position.
            y (float): Initial y position.
        """
        self.kf.x = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        self.kf.P = np.eye(4, dtype=np.float32) * 1000.0
        self.initialized = True

    def predict(self) -> np.ndarray:
        """
        Predict the next state of the ball.
        
        Returns:
            np.ndarray: Predicted 2D coordinates [x, y].
        """
        self.kf.predict()
        return self.kf.x[:2]

    def update(self, measurement: list) -> np.ndarray:
        """
        Update state with an observed measurement.
        
        Args:
            measurement (list): [x, y] coordinates of the observed ball.
            
        Returns:
            np.ndarray: Smoothed 2D coordinates [x, y].
        """
        if not self.initialized:
            self.initialize(measurement[0], measurement[1])
            return self.kf.x[:2]
            
        self.kf.update(np.array(measurement, dtype=np.float32))
        return self.kf.x[:2]

    def fill_gaps(self, track_history: list, max_gap: int = 15) -> list:
        """
        Perform smoothing and gap-filling on a sequence of ball coordinates.
        Uses prediction logic to fill in frames where detection was lost ('None').
        
        Args:
            track_history (list): Historical ball coordinates with missing detections represented by None.
                                  Example: [[100, 200], None, None, [120, 210]]
            max_gap (int): Maximum consecutive frames of missing detections to extrapolate before giving up.
            
        Returns:
            list: Interpolated/smoothed coordinate list.
        """
        smoothed_history = []
        consecutive_missed = 0
        
        for coords in track_history:
            if coords is not None:
                # Detection found
                consecutive_missed = 0
                if not self.initialized:
                    self.initialize(coords[0], coords[1])
                    smoothed_history.append(list(coords))
                else:
                    self.predict()
                    smoothed_val = self.update(coords)
                    smoothed_history.append(list(smoothed_val))
            else:
                # Detection lost
                if self.initialized and consecutive_missed < max_gap:
                    # Extrapolate using prediction
                    predicted_val = self.predict()
                    # Inject a small fake update towards our prediction to keep covariance stable
                    self.kf.x[:2] = predicted_val
                    smoothed_history.append(list(predicted_val))
                    consecutive_missed += 1
                else:
                    # Not initialized or too many missing frames
                    smoothed_history.append(None)
                    
        return smoothed_history

