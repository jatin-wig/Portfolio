
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Download, Play, Github, Linkedin, Mail } from 'lucide-react';

interface HeroProps {
  onStartAutoTour?: () => void;
}

const Hero: React.FC<HeroProps> = ({ onStartAutoTour }) => {
  const [textIndex, setTextIndex] = useState(0);
  const roles = ["AI/ML Enthusiast", "Problem Solver", "Full Stack Developer", "Data Scientist"];

  useEffect(() => {
    const interval = setInterval(() => {
      setTextIndex((prev) => (prev + 1) % roles.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleExploreProjects = (e: React.MouseEvent) => {
    e.preventDefault();
    const projectsSection = document.getElementById('projects');
    if (projectsSection) {
      projectsSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleDownloadResume = (e: React.MouseEvent) => {
    e.preventDefault();
    const resumeSection = document.getElementById('resume');
    if (resumeSection) {
      resumeSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden py-20 px-6">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <div className="absolute top-[-10%] right-[-5%] w-[500px] h-[500px] rounded-full bg-brand-purple/20 blur-[100px] animate-float" />
        <div className="absolute bottom-[-10%] left-[-10%] w-[600px] h-[600px] rounded-full bg-brand-pink/20 blur-[100px] animate-float" style={{ animationDelay: '2s' }} />
        <div className="absolute top-[40%] left-[20%] w-[300px] h-[300px] rounded-full bg-brand-vivid-purple/10 blur-[80px] animate-float" style={{ animationDelay: '4s' }} />
      </div>

      <div className="container mx-auto max-w-6xl flex flex-col-reverse md:flex-row items-center gap-12 md:gap-20">

        {/* Text Content */}
        <motion.div
          className="flex-1 text-center md:text-left z-10"
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="inline-block px-4 py-1.5 mb-6 rounded-full bg-purple-200 border border-purple-300 text-black font-bold text-sm"
          >
            👋 Welcome to my portfolio
          </motion.div>

          <h1 className="text-5xl md:text-7xl font-bold mb-6 tracking-tight leading-tight font-heading">
            Hi, I'm <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-purple to-brand-pink">Jatin</span>
            <br />
            <span className="text-3xl md:text-5xl text-gray-700 dark:text-gray-300 mt-2 block h-[1.2em]">
              <motion.span
                key={textIndex}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.5 }}
                className="block"
              >
                {roles[textIndex]}
              </motion.span>
            </span>
          </h1>

          <p className="text-xl text-gray-600 dark:text-gray-400 mb-8 max-w-lg mx-auto md:mx-0 leading-relaxed">
            Solving the world's problems with data and neurons.
            Transforming complex challenges into elegant, intelligent solutions.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center md:justify-start">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleExploreProjects}
              className="px-8 py-4 bg-brand-purple text-white rounded-xl font-semibold shadow-lg shadow-brand-purple/25 hover:bg-brand-vivid-purple transition-all flex items-center justify-center gap-2 group"
            >
              Explore Projects
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleDownloadResume}
              className="px-8 py-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-800 dark:text-white rounded-xl font-semibold hover:bg-gray-50 dark:hover:bg-gray-700 transition-all flex items-center justify-center gap-2"
            >
              <Download className="w-5 h-5" />
              Resume
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onStartAutoTour}
              className="px-8 py-4 bg-gradient-to-r from-gray-800 to-gray-900 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all flex items-center justify-center gap-2"
            >
              <Play className="w-5 h-5 fill-current" />
              Auto Tour
            </motion.button>
          </div>

          <div className="mt-10 flex items-center gap-6 justify-center md:justify-start text-gray-500">
            <a href="https://linkedin.com/in/jatin-wig" target="_blank" rel="noopener noreferrer" className="hover:text-brand-purple transition-colors">
              <Linkedin className="w-6 h-6" />
            </a>
            <a href="https://github.com/jatin-wig" target="_blank" rel="noopener noreferrer" className="hover:text-brand-purple transition-colors">
              <Github className="w-6 h-6" />
            </a>
          </div>
        </motion.div>

        {/* Hero Image */}
        <motion.div
          className="flex-1 flex justify-center relative"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <div className="relative w-72 h-72 md:w-96 md:h-96">
            <div className="absolute inset-0 bg-gradient-to-tr from-brand-purple to-brand-pink rounded-[60%_40%_30%_70%/60%_30%_70%_40%] animate-float opacity-20 blur-xl scale-110" />
            <img
              src="/lovable-uploads/d966cc57-e699-4c1e-9dd0-f30325e8eb1e.png"
              alt="Jatin Wig"
              className="relative w-full h-full object-cover rounded-[60%_40%_30%_70%/60%_30%_70%_40%] border-4 border-white dark:border-gray-800 shadow-2xl animate-float z-10"
              style={{ animationDuration: '8s' }}
            />

            {/* Floating Badges */}
            <motion.div
              className="absolute -right-4 top-10 bg-white dark:bg-gray-800 p-3 rounded-2xl shadow-xl z-20 flex items-center gap-3"
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            >
              <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center text-2xl">🤖</div>
              <div>
                <p className="text-xs text-gray-500 font-semibold">Expertise</p>
                <p className="text-sm font-bold text-gray-800 dark:text-white">GenAI</p>
              </div>
            </motion.div>

            <motion.div
              className="absolute -left-8 bottom-20 bg-white dark:bg-gray-800 p-3 rounded-2xl shadow-xl z-20 flex items-center gap-3"
              animate={{ y: [0, 10, 0] }}
              transition={{ duration: 5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
            >
              <div className="w-10 h-10 bg-purple-100 rounded-full flex items-center justify-center text-2xl">⚡</div>
              <div>
                <p className="text-xs text-gray-500 font-semibold">Approach</p>
                <p className="text-sm font-bold text-gray-800 dark:text-white">Problem-Solving at Speed</p>
              </div>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Hero;
