
import React from 'react';
import { motion } from 'framer-motion';

const tools = [
  { name: "Python", icon: "🐍" },
  { name: "VSCode", icon: "💻" },
  { name: "GitHub", icon: "🐙" },
  { name: "Jupyter", icon: "📊" },
  { name: "TensorFlow", icon: "🧠" },
  { name: "Scikit-learn", icon: "🧪" }
];

const ToolsSection = () => {
  return (
    <>
      <h3 className="text-2xl font-bold mb-6">Tools I Love</h3>
      <div className="grid grid-cols-3 gap-6">
        {tools.map((tool, index) => (
          <motion.div 
            key={index} 
            className="flex flex-col items-center"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ 
              duration: 0.5, 
              delay: index * 0.1,
              type: "spring",
              stiffness: 100,
              damping: 10
            }}
            viewport={{ once: true }}
            whileHover={{ 
              scale: 1.1, 
              rotate: [0, 5, -5, 0],
              transition: { duration: 0.5 }
            }}
          >
            <span className="text-3xl mb-2 transform transition-all duration-300">{tool.icon}</span>
            <span className="font-medium">{tool.name}</span>
          </motion.div>
        ))}
      </div>
    </>
  );
};

export default ToolsSection;