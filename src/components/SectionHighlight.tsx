
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface SectionHighlightProps {
  isVisible: boolean;
  title: string;
  description: string;
}

// Generate dynamic section descriptions that include live project data from DOM
const generateSectionDescriptions = (): Record<string, string> => {
  const staticDescriptions: Record<string, string> = {
    'Home': 'Meet Jatin - AI/ML Enthusiast & Problem Solver',
    'Skills': 'Technical superpowers and expertise areas',
    'Why Choose Me': 'What makes me unique as a developer',
    'Projects Overview': 'Innovative projects and technical achievements',
    'Research & Publications': 'Academic contributions and research work',
    'Experience': 'Professional journey and virtual internships',
    'Volunteering': 'Community involvement and social impact',
    'Education': 'Academic background and learning foundation',
    'Resume': 'Complete professional profile and achievements',
    'Contact': 'Let\'s connect and collaborate'
  };

  // Generate descriptions for all projects dynamically from rendered DOM
  const projectDescriptions: Record<string, string> = {};

  // Read from the actual rendered Projects component
  const projectCards = document.querySelectorAll('[data-project-id]');
  projectCards.forEach(card => {
    const titleElement = card.querySelector('h3'); // CardTitle
    const descriptionElement = card.querySelector('p'); // CardDescription

    if (titleElement && descriptionElement) {
      const title = titleElement.textContent?.trim();
      const description = descriptionElement.textContent?.trim();

      if (title && description) {
        projectDescriptions[title] = description;
      }
    }
  });

  return { ...staticDescriptions, ...projectDescriptions };
};

const sectionDescriptions = generateSectionDescriptions();

const SectionHighlight: React.FC<SectionHighlightProps> = ({ isVisible, title, description }) => {
  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: -50, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -50, scale: 0.9 }}
          transition={{
            type: "spring",
            damping: 20,
            stiffness: 300,
            duration: 0.6
          }}
          className="fixed top-24 md:top-auto md:bottom-8 left-4 md:left-1/2 transform md:-translate-x-1/2 z-[60] w-[85%] max-w-sm"
        >
          <div className="bg-purple-50/95 dark:bg-purple-950/90 backdrop-blur-md rounded-xl shadow-2xl border border-purple-200/50 dark:border-purple-800/50 p-3 md:p-6 text-left md:text-center">
            <motion.h3
              className="text-base md:text-xl font-bold text-gray-800 dark:text-white mb-1 md:mb-2"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
            >
              {title}
            </motion.h3>
            <motion.p
              className="text-xs md:text-base text-gray-600 dark:text-gray-400"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              {description}
            </motion.p>
            <motion.div
              className="mt-2 md:mt-3 h-1 bg-gradient-to-r from-brand-purple to-brand-vivid-purple rounded-full"
              initial={{ width: 0 }}
              animate={{ width: "100%" }}
              transition={{ delay: 0.6, duration: 1 }}
            />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export { SectionHighlight, sectionDescriptions };