import React from 'react';
import { motion } from 'framer-motion';
import { GraduationCap } from 'lucide-react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const educationHistory = [
  {
    id: 1,
    institution: "Birla Institute of Technology And Science - Pilani",
    degree: "Bachelor of Science in Computer Science",
    description: "Focused on artificial intelligence, machine learning, and software engineering. Completed coursework in data structures, algorithms, database systems, and web development.",
    coursework: ["Artificial Intelligence", "Machine Learning", "Data Structures", "Software Engineering", "Database Systems"]
  },
  {
    id: 2,
    institution: "Bharat National Public School",
    degree: "High School",
    description: "Science stream (Physics, Chemistry, Maths) with Computer Science. Active participant in coding competitions and science fairs.",
    coursework: ["Physics", "Chemistry", "Mathematics", "Computer Science"]
  }
];

const Education = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.3
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        type: "spring",
        damping: 12,
        stiffness: 100
      }
    }
  };

  return (
    <section className="py-16 px-6 bg-gradient-to-b from-gray-50 to-white">
      <div className="max-w-6xl mx-auto">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="text-center mb-16"
        >
          <motion.h2
            variants={itemVariants}
            className="text-4xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-gray-800 to-brand-purple"
          >
            Education
          </motion.h2>
          <motion.p
            variants={itemVariants}
            className="text-gray-600 text-lg max-w-3xl mx-auto"
          >
            Academic foundation and continuous learning journey in technology and computer science.
          </motion.p>
        </motion.div>

        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
          className="space-y-8"
        >
          {educationHistory.map((education, index) => (
            <motion.div
              key={education.id}
              variants={itemVariants}
              custom={index}
            >
              <Card className="overflow-hidden hover:shadow-xl transition-all duration-500 border-none shadow-lg group">
                <CardHeader className="bg-gradient-to-r from-brand-light-purple/20 to-brand-purple/10 pb-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-brand-purple/10 rounded-full">
                        <GraduationCap className="h-6 w-6 text-brand-purple dark:text-purple-300" />
                      </div>
                      <div>
                        <CardTitle className="text-2xl leading-tight text-gray-800">
                          {education.institution}
                        </CardTitle>
                        <p className="text-brand-purple dark:text-purple-300 font-medium text-lg mt-1">
                          {education.degree}
                        </p>
                      </div>
                    </div>
                  </div>
                </CardHeader>

                <CardContent className="pt-6">
                  <p className="text-gray-600 leading-relaxed mb-6">
                    {education.description}
                  </p>

                  <div>
                    <h4 className="font-semibold text-gray-800 mb-3">Key Coursework</h4>
                    <div className="flex gap-2 flex-wrap">
                      {education.coursework.map((course) => (
                        <motion.span
                          key={course}
                          className="px-3 py-1 bg-gradient-to-r from-blue-100 to-blue-50 text-blue-700 text-sm rounded-full border border-blue-200"
                          whileHover={{
                            scale: 1.05,
                            backgroundColor: "#eff6ff"
                          }}
                          transition={{ type: "spring", stiffness: 400, damping: 10 }}
                        >
                          {course}
                        </motion.span>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default Education;
