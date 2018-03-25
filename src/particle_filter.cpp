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
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	//initialize to 500 according to the map size 50
	num_particles = 500;
	
	default_random_engine gen;
	//creates normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(x, std[0]);
	//creates normal (Gaussian) distribution for y.
	normal_distribution<double> dist_y(y, std[1]);
	//creates normal (Gaussian) distribution for theta.
	normal_distribution<double> dist_theta(theta, std[2]);

	// cout<<"=======Start initializing=========\n";
	for(int i=0;i<num_particles;++i){
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		weights.push_back(1.0);
		particles.push_back(p);
		// cout<<"id: "<<p.id<<"\tx: "<<p.x<<"\ty: "<<p.y<<"\ttheta: "<<p.theta<<"\n";
	}
	is_initialized = true;
	// cout<<"Particles initialization succeeded!\n";
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	// cout<<"=======Start prediction=========\n";
	for(int i = 0; i<particles.size(); ++i){
		//creates normal (Gaussian) distribution for x.
		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		//creates normal (Gaussian) distribution for y.
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		//creates normal (Gaussian) distribution for theta.
		normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
		//yaw rate is zero
		if(yaw_rate <= 0.000001){
			particles[i].x = particles[i].x + velocity*delta_t*cos(particles[i].theta) + dist_x(gen);
			particles[i].x = particles[i].y + velocity*delta_t*sin(particles[i].theta) + dist_y(gen);
			particles[i].theta += dist_theta(gen);
		}else{
			particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta)) + dist_x(gen);
			particles[i].x += (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t)) + dist_x(gen);
			particles[i].theta += yaw_rate*delta_t + dist_theta(gen);
		}
		// cout<<"id: "<<particles[i].id<<"\tx: "<<particles[i].x<<"\ty: "<<particles[i].y<<"\ttheta: "<<particles[i].theta<<"\n";
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for(int i = 0 ; i < particles.size(); ++i){
		//coordinate-axis transformation
		std::vector<LandmarkObs> predicted;

		//go through all landmark observations and transform
		for(int j = 0; j<observations.size();++j){
			LandmarkObs lmobs;
			lmobs.x = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
			lmobs.x = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
			predicted.push_back(lmobs);
		}

		std::vector<int> ids;
		std::vector<double> xs;
		std::vector<double> ys;

		// calculate normalization term
		double gauss_norm= (1/(2 * M_PI * std_landmark[0] * std_landmark[1]));

		for(int k = 0; k < predicted.size(); ++k){
			//assign initial distance to the map range
			double n_dist =  sensor_range;
			int n_id;
			double n_x;
			double n_y;
			//go over all landmarks on the map to find the nearest and assign the id
			for(int n = 0; n< map_landmarks.landmark_list.size(); ++n){
				double j_dist = dist(predicted[k].x, predicted[k].y, map_landmarks.landmark_list[n].x_f, map_landmarks.landmark_list[n].y_f);
				//if current observed landmark is the nearest then update
				if(j_dist <= n_dist){
					n_dist = j_dist;
					n_x = map_landmarks.landmark_list[n].x_f;
					n_y = map_landmarks.landmark_list[n].y_f;
					n_id = map_landmarks.landmark_list[n].id_i;
				}
			}
			ids.push_back(n_id);
			xs.push_back(n_x);
			ys.push_back(n_y);

			//update weights for current particle using the result from previous association
			// calculate exponent
			double exponent= pow((predicted[k].x - n_x),2)/(2 * std_landmark[0] * std_landmark[0]) + pow((predicted[k].y - n_y), 2)/(2 * std_landmark[1] * std_landmark[1]);
			double wk = gauss_norm * exp(-exponent);
			particles[i].weight *=wk;

		}

		SetAssociations(particles[i], ids, xs, ys);
		// //association
		// dataAssociation(predicted,map_landmarks);
		predicted.clear();
		ids.clear();
		xs.clear();
		ys.clear();
		weights.clear();
		weights.push_back(particles[i].weight);
		cout<<"Weight of "<<i<<"-th particle: "<<particles[i].weight<<"\n";
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
