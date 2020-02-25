from src.Seeker import Seeker
from src.Detector import Detector
from src.DSBuilder import DSBuilder
import keras 
def main():
        
    # detector = Detector()
    # detector.getFaces()

    dataset_dir = DSBuilder()
    dataset_dir.seek()

    print(dataset_dir.dataset)

if __name__ == '__main__':
    main()