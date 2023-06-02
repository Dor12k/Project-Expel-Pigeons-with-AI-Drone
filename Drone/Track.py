
# Class represent a track which is array od path objects
class Track:

    # track = []

    def __init__(self):
        self.track = []

    @property
    def track(self):
        return self.__track

    @track.setter
    def track(self, drone_track):
        self.__track = drone_track

    # Get Path object
    def add_path(self, path):
        self.track.append(path)

    # Get track length
    def get_length(self):
        return len(self.track)

    # Print the track
    def print_track(self):

        for path in self.track:
            if path.get_direction() == -1:
                print("Stand")
            elif path.get_direction() == 0:
                if 0 < path.get_speed():
                    print("Right: ", path.get_distance(), "cm")
                else:
                    print("Left: ", path.get_distance(), "cm")
            elif path.get_direction() == 1:
                if 0 < path.get_speed():
                    print("Forward: ", path.get_distance(), "cm")
                else:
                    print("Backward: ", path.get_distance(), "cm")
            elif path.get_direction() == 2:
                if 0 < path.get_speed():
                    print("Up: ", path.get_distance(), "cm")
                else:
                    print("Down: ", path.get_distance(), "cm")
            elif path.get_direction() == 3:
                if 0 < path.get_speed():
                    print("Rotate Right: ", path.get_distance(), "degree")
                else:
                    print("Rotate Left: ", path.get_distance(), "degree")

    # Return track objects
    def __repr__(self):
        return f"fTrack: {self.track}"

    # Return string from str method in Path class
    def __str__(self):

        track = "Track length: " + str(Track.get_length(self))

        for i, path in enumerate(self.track):
            track = track + "\n" + str(i) + ":   " + str(path)

        return track
