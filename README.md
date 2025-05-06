# Capacitive Routing (Bachelor's Thesis Project)

This repository contains the routing algorithm implementation developed as part of my bachelor's thesis on capacitive keys. The routing system generates internal conductive paths within 3D-printed tangible objects, designed for secure interaction with capacitive touchscreens.

## Overview

The goal of this routing algorithm is to:
- Connect predefined capacitive points inside of a tangible object
- Avoid capacitive bridging by enforcing a minimum distance between routed paths
- Optimize path layout using a modified A* algorithm within a voxel grid
- Support export of generated paths for multi-material 3D printing

## Features

- A* pathfinding in 3D voxel space
- Path collision avoidance and spacing enforcement
- Export as polyline sequences or STL-compatible geometry


