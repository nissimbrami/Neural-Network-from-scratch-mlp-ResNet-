from scipy.io import loadmat
import numpy as np
import os


class DataLoader:
    def __init__(self, dataset_path):
        """
        טוען נתונים מקובץ MAT

        Args:
            dataset_path: נתיב לקובץ ה-MAT
        """
        self.data = loadmat(dataset_path)
        self.dataset_name = os.path.basename(dataset_path).replace('Data.mat', '')
        print(f"\nבדיקת נתונים גולמיים:")
        for key in ['Yt', 'Ct', 'Yv', 'Cv']:
            if key in self.data:
                print(f"{key} shape: {self.data[key].shape}")

    def get_data(self):
        """
        מחזיר את נתוני האימון והבדיקה

        Returns:
            tuple: ((X_train, y_train), (X_test, y_test))
        """
        # טעינת נתונים
        X_train = self.data['Yt'].T
        X_test = self.data['Yv'].T

        # זיהוי סוג הדאטאסט
        if 'GMM' in self.dataset_name:
            print("זוהה כדאטאסט GMM")
            y_train = np.argmax(self.data['Ct'], axis=0)
            y_test = np.argmax(self.data['Cv'], axis=0)
        elif 'Peaks' in self.dataset_name:
            print("זוהה כדאטאסט Peaks")
            y_train = np.argmax(self.data['Ct'], axis=0)
            y_test = np.argmax(self.data['Cv'], axis=0)
        else:  # SwissRoll
            print("זוהה כדאטאסט SwissRoll")
            y_train = self.data['Ct'].ravel()
            y_test = self.data['Cv'].ravel()

            # תיקון המימדים אם צריך
            if y_train.shape[0] == 2 * X_train.shape[0]:
                print("\nמתקן כפילות במימדי y_train...")
                y_train = y_train[:X_train.shape[0]]
            if y_test.shape[0] == 2 * X_test.shape[0]:
                print("מתקן כפילות במימדי y_test...")
                y_test = y_test[:X_test.shape[0]]

        # המרת labels לint
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

        print("\nמידע על הדאטאסט:")
        print(f"צורת נתוני אימון: X: {X_train.shape}, y: {y_train.shape}")
        print(f"צורת נתוני בדיקה: X: {X_test.shape}, y: {y_test.shape}")
        print(f"מחלקות ייחודיות באימון: {np.unique(y_train)}")
        print(f"טווח ערכים בX: [{X_train.min():.2f}, {X_train.max():.2f}]")

        # בדיקה סופית
        print("\nצורות סופיות:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, y_train type: {y_train.dtype}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}, y_test type: {y_test.dtype}")

        return (X_train, y_train), (X_test, y_test)