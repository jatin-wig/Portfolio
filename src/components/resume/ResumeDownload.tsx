
import React from 'react';
import { motion } from 'framer-motion';

const ResumeDownload = () => {
  return (
    <motion.a 
      href="https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz123456/view?usp=sharing" 
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-3 px-6 py-4 bg-gradient-to-r from-brand-purple to-brand-vivid-purple text-white rounded-lg hover:shadow-lg transition-all duration-300 mb-12"
      whileHover={{ scale: 1.05, boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)" }}
      whileTap={{ scale: 0.95 }}
    >
      <span className="text-white text-2xl">📄</span>
      <div>
        <div className="font-bold">Download Resume</div>
        <div className="text-sm opacity-80">PDF Format (2.4 MB)</div>
      </div>
    </motion.a>
  );
};

export default ResumeDownload;
