/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 50;

	particles.resize(num_particles);

	default_random_engine gen;
	normal_distribution<double> N_x_init(x, std[0]);
	normal_distribution<double> N_y_init(y, std[1]);
	normal_distribution<double> N_theta_init(theta, std[2]);

	std::cout << N_x_init(gen) << " " << N_y_init(gen) << " " << N_theta_init(gen) << '\n';

	for (unsigned int i = 0; i < num_particles; i++) {
		particles[i].id = i + 1;
		particles[i].x = N_x_init(gen);
		particles[i].y = N_y_init(gen);
		particles[i].theta = N_theta_init(gen);
		particles[i].weight = 1.0;
	}

	is_initialized = true;

	std::cout << "Num Particles " << num_particles << '\n';
	std::cout << "Initialized." << '\n';
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	normal_distribution<double> N_x_init(0, std_pos[0]);
	normal_distribution<double> N_y_init(0, std_pos[1]);
	normal_distribution<double> N_theta_init(0, std_pos[2]);

	if (fabs(yaw_rate) > 0.00001) { // avoid deviding by zero
		for (unsigned int i = 0; i < num_particles; i++) {
			particles[i].x = N_x_init(gen) + particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y = N_y_init(gen) + particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta = N_theta_init(gen) + particles[i].theta + yaw_rate * delta_t;
		}
	} else {
		for (unsigned int i = 0; i < num_particles; i++) {
			particles[i].x = N_x_init(gen) + particles[i].x + velocity * delta_t * cos(particles[i].theta);
			particles[i].y = N_y_init(gen) + particles[i].y + velocity * delta_t * sin(particles[i].theta);
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {

	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	double C = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
	double stdx_2 = 2.0 * pow(std_landmark[0], 2);
	double stdy_2 = 2.0 * pow(std_landmark[1], 2);

	for (unsigned int i = 0; i < num_particles; i++) { // iterate through each particle
		// iterate through each CAR sensor observation in Vehicle Coordinate system

		for (unsigned int z = 0; z < observations.size(); z++) {
			double obs_to_map_x = observations[z].x * cos(particles[i].theta) - observations[z].y * sin(particles[i].theta) + particles[i].x;
			double obs_to_map_y = observations[z].x * sin(particles[i].theta) + observations[z].y * cos(particles[i].theta) + particles[i].y;

			double max_dist = sensor_range;
			double closest_mark_x;
			double closest_mark_y;

			for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); k++) {
				double l_dist = dist(obs_to_map_x, obs_to_map_y,
									 map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);

				if (l_dist < max_dist){
					closest_mark_x = map_landmarks.landmark_list[k].x_f;
					closest_mark_y = map_landmarks.landmark_list[k].y_f;
					max_dist = l_dist;
				}
			}

			// Multivariate Gaussians
			double prob = C * exp( - (pow(obs_to_map_x - closest_mark_x, 2) / stdx_2 + pow(obs_to_map_y - closest_mark_y, 2) / stdy_2));

			particles[i].weight *= prob;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> resampled_particles;

	// put all particles.weight into the weights vector
	for (size_t i = 0; i < num_particles; i++)
	{
		weights.push_back(particles[i].weight);
	}
	default_random_engine gen;

	// distribute the weights randomly from weights.begin to
	// weights. end using discrete_distribution function

	// std::discrete_distribution produces random integers on the interval [0, n),
	// where the probability of each individual integer i is defined as w
	// i/S, that is the weight of the ith integer divided by the sum of all n weights.
	// std::discrete_distribution satisfies all requirements of RandomNumberDistribution

	discrete_distribution<int> weight_distribution(weights.begin(), weights.end());

	for (int i = 0; i < num_particles; i++) {
		int weighted_index = weight_distribution(gen);
		// what is range of weighted_index
		resampled_particles.push_back(particles[weighted_index]);
	}

	particles = resampled_particles;
	weights.clear();
	for (int i = 0; i < num_particles; i++) {
		particles[i].weight = 1.;
	}

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
