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
	for(int i=0;i<num_particles;i++){
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
	for(int i = 0; i<particles.size(); i++){
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
	// cout<<"Start dataAssociation!!\n";

	for(int k = 0; k < observations.size(); k++)
	{
		//assign initial distance to the map range
		double nearest_dist =  numeric_limits<double>::max();
		int nearest_partical_id = -1;
		//go over all landmarks on the map to find the nearest and assign the id
		for(int n = 0; n< predicted.size(); n++)
		{
			double dist_n = dist(observations[k].x, observations[k].y, predicted[n].x, predicted[n].y);
			//if current observed landmark is the nearest then update
			if(dist_n < nearest_dist)
			{
				nearest_dist = dist_n;
				nearest_partical_id = predicted[n].id;
			}
		}
		//assign the current nearest landmark id on the map to the current observation
		if(nearest_partical_id > 0){
			observations[k].id = nearest_partical_id;
			// cout<<"nearest dist:\t"<<nearest_dist<<"\twith id:\t"<<nearest_partical_id<<"\n";
			// cout<<"\tobs_id:\t"<<observations[k].id<<"\tobs_x:\t"<<observations[k].x<<"\tobs_y:\t"<<observations[k].y<<"\n";
			// cout<<"\tmap_id:\t"<<predicted[nearest_partical_id].id<<"\tmap_x:\t"<<predicted[nearest_partical_id].x<<"\tmap_y:\t"<<predicted[nearest_partical_id].y<<"\n";
		}
	}
	// cout<<"Finished dataAssociation!!\n";
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

	// cout<<"==========start to update==============\n";
	// weights.clear();
	// cout<<"finished assign predicted vector!\n";

	for(int i = 0 ; i < particles.size(); i++)
	{

			// get landmarks from map
			vector<LandmarkObs> predicted;
			Map::single_landmark_s slandmark;
			for(int i = 0;i < map_landmarks.landmark_list.size(); i++)
			{
				slandmark = map_landmarks.landmark_list[i];
				double x_0 = slandmark.x_f;
				double y_0 = slandmark.y_f;
				double dist_to_particle = dist(x_0, y_0, particles[i].x, particles[i].y);
				if(dist_to_particle < sensor_range)
				{
					LandmarkObs lm_i;
					lm_i.id = slandmark.id_i;
					lm_i.x = x_0;
					lm_i.y = y_0;
					predicted.push_back(lm_i);
				}
			}

		vector<LandmarkObs> obs_global;
		//go through all landmark observations and transform coordinate-axis to global map axis
		for(int j = 0; j<observations.size();j++)
		{
			LandmarkObs lm_global;
			lm_global.id = -1;
			lm_global.x = particles[i].x + cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y;
			lm_global.x = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
			obs_global.push_back(lm_global);
		}
		// cout<<"Finished coordinate-axis transformation!!\n";
		// association
		// cout<<"start dataAssociation\n";
		dataAssociation(predicted, obs_global);
		// cout<<"finished dataAssociation\n";

		std::vector<int> ids;
		std::vector<double> xs;
		std::vector<double> ys;

		particles[i].weight = 1.0;
		weights[i] = 1.0;
		// cout<<"starting weight:"<<particles[i].weight<<"\n";
		//update weights for current particle using the result from previous association
		double gauss_norm = 1/(2 * M_PI * std_landmark[0] * std_landmark[1]);
		double delta_x_2 = std_landmark[0]*std_landmark[0];
		double delta_y_2 = std_landmark[1]*std_landmark[1];
		// cout<<"gauss_norm: "<<gauss_norm<<"\tstd_x:\t"<<std_landmark[0]<<"\tstd_y:\t"<<std_landmark[1]<<"\n";
		// calculate exponent
		for(int n = 0; n<obs_global.size(); n++)
		{

			int id_m = obs_global[n].id;
			// cout<<"calculating for id: "<<id_m<<"\tid: "<<obs_global[n].id<<"\tx: "<<obs_global[n].x<<"\ty: "<<obs_global[n].y<<"\n";
			for(int m = 0; m < predicted.size(); m++)
			{
				if(id_m == predicted[m].id)
				{
					double diff_x = obs_global[n].x - predicted[id_m-1].x;
					double diff_y = obs_global[n].y - predicted[id_m-1].y;
					double exponent= diff_x*diff_x/(2*delta_x_2) + diff_y*diff_y/(2*delta_y_2);
					double wk = gauss_norm * exp(-exponent);
					if(wk > 0.0){
						particles[i].weight *=wk;
					}
					// break;

					// if(diff_x > sensor_range || diff_y > sensor_range){
					// 	continue;
					// }
					// cout<<id_m<<"-th accumulated weight:\t"<<wk<<"\n";
					// if(wk > 0.0)
					// {
					
					// }
				}
			}


			ids.push_back(id_m);
			xs.push_back(obs_global[n].x);
			ys.push_back(obs_global[n].y);

		}

		if(particles[i].weight >= 1.0)
		{
			particles[i].weight == 0.0;
		}
		// else
		// {
		weights[i] = particles[i].weight;
		// }
		// weights[i] = particles[i].weight;
			// if(id_m == predicted[])
			// {
			// 	double diff_x = obs_global[n].x - predicted[id_m-1].x;
			// 	double diff_y = obs_global[n].y - predicted[id_m-1].y;

			// 	// if(diff_x > sensor_range || diff_y > sensor_range){
			// 	// 	continue;
			// 	// }
			// 	double exponent= diff_x*diff_x/(2*delta_x_2) + diff_y*diff_y/(2*delta_y_2);

			// 	double wk = gauss_norm * exp(-exponent);
			// 	// cout<<id_m<<"-th accumulated weight:\t"<<wk<<"\n";
			// 	if(wk > 0.0)
			// 	{
			// 		ids.push_back(id_m);
			// 		xs.push_back(obs_global[n].x);
			// 		ys.push_back(obs_global[n].y);

			// 		particles[i].weight *=wk;
			// 	}
			// 	// cout<<id_m<<"-th diff_x:\t"<<diff_x<<"\tdiff_y:\t"<<diff_y<<"\texponent part:\t"<<exponent<<"\texponent\t"<<exp(-exponent)<<"\tpar_w:\t"<<particles[i].weight<<"\twk:\t"<<wk<<"\n";
			// }
			// cout<<id_m<<"-th accumulated weight:\t"<<particles[i].weight <<"\n";
		// }

		// if(particles[i].weight == 1.0)
		// {
		// 	particles[i].weight = 0.0;
		// }
		
		// if(particles[i].weight > 0.0)
		// {
			// cout<<"starting calculate weight for :"<<particles[i].id<<endl;
			SetAssociations(particles[i], ids, xs, ys);
			// cout<<"finished calculate weight for :"<<particles[i].id<<endl;
		// }
		// weights.push_back(particles[i].weight);
			cout<<"Weight of "<<i<<"-th particle: "<<particles[i].weight<<"\n";
		// calculate normalization term

	}

	//normalize weights for particles
	// double weights_sum = 0;
	// for(int k = 0; k < particles.size();k++)
	// {
	// 	weights_sum += particles[k].weight;
	// }
	
	// cout<<"Total weights: "<<weights_sum<<"\n";
	// // weights.clear();
	// for(int i = 0; i<particles.size(); i++)
	// {
	// 	particles[i].weight /= weights_sum;
	// 	weights[i] = particles[i].weight;
	// 	cout<<i<<"-th weight: "<<particles[i].weight<<"\n";
	// }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	// cout<<"Start resample!\n";
	random_device rd;
	// mt19937 gen(rd());
	default_random_engine gen;

	double max_w = 0;
	for( int i=0; i< particles.size();i++){
		if(particles[i].weight > max_w){
			max_w = particles[i].weight;
		}
	}

	std::vector<Particle> new_particles;
	cout<<"got max weight: "<<max_w<<"\n";
	// uniform random distribution
	uniform_real_distribution<double> unirealdist(0.0, 1.0);

	double beta = 0.0;
	int p_size = particles.size();
	int init_id = rand() % p_size;

	discrete_distribution<> d(weights.begin(), weights.end());
	//Selected particles
	// int selected[p_size];
	// //initialize elements to 0 representing not selected
	// for(int i =0 ; i< p_size;i++)
	// {
	// 	selected[i] = 0;
	// }

	for(int i = init_id; i< p_size+init_id;i++)
	{
		beta += unirealdist(gen) * 2.0 * max_w;
		double wt = weights[i];
		while ( wt < beta)
		{
			beta -= weights[i];
			i++;
			if(i >= p_size+init_id)
				break;
			wt = weights[i%p_size];
			cout<<"Drop "<<i%p_size<<"-th particle with weight: "<<weights[i%p_size]<<"\n";
		}
		new_particles.push_back(particles[i]);
		// new_particles.push_back(particles[d(gen)]);
		// // if(particles[i].weight > 0.0)
		// {
		// 	beta = beta + d(gen)/max_w;
		// 	while(particles[i].weight < beta){
		// 		beta = beta - particles[i].weight;
		// 		// particles.erase(particles.begin()+i);
		// 		i++;
		// 	}
		// 	//else take this particle
		// 	Particle  new_particle = particles[i];
		// 	new_particles.push_back(new_particle);
		// }
	}
	particles = new_particles;
	// for(int i = 0; i < p_size;i++)
	// {
	// 	if(selected[i] < 1)
	// 	{//not selected
	// 		particles.erase(particles.begin() + i - deleted_count);
	// 		deleted_count++;
	// 	}
	// }
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
