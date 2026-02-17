
import React, { useState } from 'react';
import Header from '../components/Header';
import Hero from '../components/Hero';
import Superpowers from '../components/Superpowers';
import WhyChooseMe from '../components/WhyChooseMe';
import Projects from '../components/Projects';
import Publications from '../components/Publications';
import Experience from '../components/Experience';
import Volunteering from '../components/Volunteering';
import Education from '../components/Education';
import Resume from '../components/Resume';
// import Contact from '../components/Contact'; // Removed 
import Footer from '../components/Footer';
import AutoView from '../components/AutoView';

const HomePage = () => {
  const [showAutoTour, setShowAutoTour] = useState(false);

  const handleStartAutoTour = () => {
    setShowAutoTour(true);
  };

  const handleStopAutoTour = () => {
    setShowAutoTour(false);
  };

  return (
    <div className="min-h-screen">
      {showAutoTour && (
        <AutoView
          showFullUI={true}
          onTourComplete={handleStopAutoTour}
          onTourStop={handleStopAutoTour}
          autoStart={true}
        />
      )}
      <Header />
      <section id="hero">
        <Hero onStartAutoTour={handleStartAutoTour} />
      </section>
      <section id="superpowers">
        <Superpowers />
      </section>
      <section id="why-choose-me">
        <WhyChooseMe />
      </section>
      <section id="projects">
        <Projects />
      </section>
      <section id="publications">
        <Publications />
      </section>
      <section id="experience">
        <Experience />
      </section>
      <section id="volunteering">
        <Volunteering />
      </section>
      <section id="education">
        <Education />
      </section>
      <section id="resume">
        <Resume />
      </section>
      <section id="contact" className="py-20 px-6">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          <h2 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-brand-purple to-brand-pink">
            Ready to make something amazing?
          </h2>
          <p className="text-xl text-gray-500 dark:text-gray-400 max-w-2xl mx-auto">
            I'm always looking for new challenges and interesting projects.
          </p>
          <div className="flex justify-center">
            <a
              href="/contact"
              className="group relative px-8 py-4 bg-white dark:bg-white/10 text-gray-900 dark:text-white rounded-full font-semibold text-lg overflow-hidden transition-all hover:scale-105 hover:shadow-lg hover:shadow-brand-purple/20"
            >
              <span className="relative z-10 flex items-center gap-2">
                Get in Touch
                <span className="group-hover:translate-x-1 transition-transform">→</span>
              </span>
              <div className="absolute inset-0 bg-gradient-to-r from-brand-purple to-brand-pink opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            </a>
          </div>
        </div>
      </section>
      <Footer />
    </div>
  );
};

export default HomePage;
