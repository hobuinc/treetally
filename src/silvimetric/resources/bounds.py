import json
import ast

#TODO should these bounds have a buffer?
class Bounds(dict): #for JSON serializing
    """Simple class to represent a 2 or 3-dimensional bounding box that can be
    generated from both JSON or PDAL bounds form."""
    def __init__(self, minx: float, miny: float, maxx: float, maxy: float):
        self.minx = float(minx)
        """minimum X Plane"""
        self.miny = float(miny)
        """minimum Y plane"""
        self.maxx = float(maxx)
        """maximum X plane"""
        self.maxy = float(maxy)
        """maximum Y plane"""

    def __eq__(self, other):
        return  other.minx == self.minx and \
                other.miny == self.miny and \
                other.maxx == self.maxx and \
                other.maxy == self.maxy
    def __ne__(self, other):
        return not other.__eq__(self)

    def __bool__(self):
        return True

    def __repr__(self) -> str:
        return str(self.get())

    @staticmethod
    def from_string(bbox_str: str):
        """Create Bounds object from a PDAL bounds string in the form:

        "([1,101],[2,102],[3,103])"
        "{\"minx\": 1,\"miny\": 2,\"maxx\": 101,\"maxy\": 102}"
        "[1,2,101,102]"
        "[1,2,3,101,102,103]"

        :param bbox_str: Bounds string
        :raises Exception: Unable to load Bounds via json or PDAL bounds type
        :raises Exception: Bounding boxes must have either 4 or 6 elements
        :return: Bounds object
        """

        try:
            bbox = json.loads(bbox_str)
        except json.decoder.JSONDecodeError as e:
            # try manual parsing of PDAL bounds
            # ([1,101],[2,102],[3,103])
            bbox_str = bbox_str.strip()
            if bbox_str[0] != '(':
                raise Exception(f"Unable to load Bounds via json or PDAL bounds type {e}")
            t = ast.literal_eval(bbox_str)
            minx = t[0][0]
            maxx = t[0][1]
            miny = t[1][0]
            maxy = t[1][1]
            return Bounds(minx, miny, maxx, maxy)


        # parse explicit style
        if 'minx' in bbox:
            minx = float(bbox['minx'])
            miny = float(bbox['miny'])
            maxx = float(bbox['maxx'])
            maxy = float(bbox['maxy'])
            return Bounds(minx, miny, maxx, maxy)

        # parse GeoJSON array style
        if len(bbox) == 4:
            return Bounds(float(bbox[0]), float(bbox[1]), float(bbox[2]),
                            float(bbox[3]))
        elif len(bbox) == 6:
            return Bounds(float(bbox[0]), float(bbox[1]), float(bbox[3]),
                            float(bbox[4]))
        else:
            raise Exception("Bounding boxes must have either 4 or 6 elements")


    def get(self) -> list[float]:
        """Return Bounds as a list of floats

        :return: list of floats in form [minx, miny, maxx, maxy]
        """
        return [self.minx, self.miny, self.maxx, self.maxy]


    def to_string(self) -> str:
        """Return string representation of Bounds

        :return: string of a list of floats in form [minx, miny, maxx, maxy]
        """
        return self.__repr__()

    def to_json(self) -> list[float]:
        """Return object as a json serializable list

        :return: list of floats in form [minx, miny, maxx, maxy]
        """
        return self.get()


    def bisect(self):
        """Bisects the current Bounds

        :yield: 4 child bounds
        """
        centerx = self.minx + ((self.maxx - self.minx)/ 2)
        centery = self.miny + ((self.maxy - self.miny)/ 2)
        yield from [
            Bounds(self.minx, self.miny, centerx, centery), # lower left
            Bounds(centerx, self.miny, self.maxx, centery), # lower right
            Bounds(self.minx, centery, centerx, self.maxy), # top left
            Bounds(centerx, centery, self.maxx, self.maxy), # top right
        ]

    def disjoint(self, other):
        """Determine if two bounds are disjointed

        :param other: Bounds this object is being compared to
        :return: True if this box shares no point with the other box, otherwise False
        """
        if other.minx > self.maxx or other.maxx < self.minx:
            return True
        if other.miny > self.maxy or other.maxy < self.miny:
            return True
        return False

    def adjust_alignment(self, resolution, alignment):
        if alignment.lower() == "pixelisarea":
            xmindif = self.minx % resolution
            xmaxdif = self.maxx % resolution
            ymaxdif = self.maxy % resolution
            ymindif = self.miny % resolution

            if xmindif:
                self.minx = self.minx - xmindif
            if xmaxdif:
                self.maxx = self.maxx + (resolution - xmaxdif)
            if ymindif:
                self.miny = self.miny - ymindif
            if ymaxdif:
                self.maxy = self.maxy + (resolution - ymaxdif)
        elif alignment.lower() == "pixelispoint":
            xmindif = self.minx % resolution
            xmaxdif = self.maxx % resolution
            ymaxdif = self.maxy % resolution
            ymindif = self.miny % resolution

            if xmindif > resolution / 2.0:
                self.minx -= (xmindif - resolution / 2.0)
            elif xmindif < resolution / 2.0:
                self.minx -= (xmindif + resolution / 2.0)
            else:
                pass
            if ymindif > resolution / 2.0:
                self.miny -= (ymindif - resolution / 2.0)
            elif ymindif < resolution / 2.0:
                self.miny -= (ymindif + resolution / 2.0)
            else:
                pass
            if xmaxdif > resolution / 2.0:
                self.maxx += resolution / 2.0 + (resolution - xmaxdif)
            elif xmaxdif < resolution / 2.0:
                self.maxx += resolution / 2.0 - xmaxdif
            else:
                pass
            if ymaxdif > resolution / 2.0:
                self.maxy += resolution / 2.0 + (resolution - ymaxdif)
            elif ymaxdif < resolution / 2.0:
                self.maxy += resolution / 2.0 - ymaxdif
            else:
                pass
        else:
            raise Exception(f"Invalid pixel alignment: {alignment}")

    @staticmethod
    def shared_bounds(first, second):
        """Find the Bounds that is shared between two Bounds.

        :param first: First Bounds object for comparison.
        :param second: Second Bounds object for comparison.
        :returns: None if there is no overlap, otherwise the shared Bounds
        """

        if first.disjoint(second):
            return None

        minx = max(first.minx, second.minx)
        maxx = min(first.maxx, second.maxx)
        miny = max(first.miny, second.miny)
        maxy = min(first.maxy, second.maxy)

        return Bounds(minx, miny, maxx, maxy)
