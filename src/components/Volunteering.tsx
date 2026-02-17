import React from 'react';
import { motion } from 'framer-motion';
import { Heart, Users, Calendar, MapPin } from 'lucide-react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const volunteeringExperiences = [
  {
    id: 1,
    organization: "Google Developer Group Abu Dhabi & Sharjah",
    role: "Organiser",
    location: "Abu Dhabi, UAE",
    period: "May 2025 - Present",
    description: "Contributing to the tech community by organizing events, workshops, and knowledge sharing sessions. Helping developers connect and learn about Google technologies and best practices.",
    activities: ["Event Organization", "Workshop Facilitation", "Community Building", "Tech Mentoring"],
    icon: Users
  },
  {
    id: 2,
    organization: "TechSpark - Bharat National Public School",
    role: "Leading Technical Member",
    location: "Delhi, India",
    period: "2022",
    description: "Organized events, made videos for them, and helped students with technical stuff. Mentoring students in technology and programming, organizing coding workshops, and inspiring the next generation of developers and innovators.",
    activities: ["Event Organization", "Video Production", "Student Mentoring", "Technical Support"],
    icon: Heart
  }
];

const Volunteering = () => {
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
    <section className="py-16 px-6 bg-gradient-to-b from-white to-gray-50">
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
            Volunteering & Community
          </motion.h2>
          <motion.p
            variants={itemVariants}
            className="text-gray-600 text-lg max-w-3xl mx-auto"
          >
            Giving back to the community through technology education and developer advocacy.
          </motion.p>
        </motion.div>

        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
          className="grid gap-8 md:grid-cols-2"
        >
          {volunteeringExperiences.map((experience, index) => {
            const IconComponent = experience.icon;
            return (
              <motion.div
                key={experience.id}
                variants={itemVariants}
                custom={index}
              >
                <Card className="overflow-hidden hover:shadow-xl transition-all duration-500 border-none shadow-lg group h-full">
                  <CardHeader className="bg-gradient-to-r from-brand-light-purple/20 to-brand-purple/10 pb-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-purple-200 rounded-full">
                          <IconComponent className="h-6 w-6 text-black" />
                        </div>
                        <div>
                          <CardTitle className="text-xl leading-tight text-gray-800">
                            {experience.organization}
                          </CardTitle>
                          <p className="text-brand-purple font-medium text-sm mt-1">
                            {experience.role}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-4 text-sm text-gray-600">
                      <div className="flex items-center gap-1">
                        <MapPin className="h-4 w-4" />
                        <span>{experience.location}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Calendar className="h-4 w-4" />
                        <span>{experience.period}</span>
                      </div>
                    </div>
                  </CardHeader>

                  <CardContent className="pt-6">
                    <p className="text-gray-600 leading-relaxed mb-6">
                      {experience.description}
                    </p>

                    <div>
                      <h4 className="font-semibold text-gray-800 mb-3">Key Activities</h4>
                      <div className="flex gap-2 flex-wrap">
                        {experience.activities.map((activity) => (
                          <motion.span
                            key={activity}
                            className="px-3 py-1 bg-gradient-to-r from-gray-100 to-gray-50 text-gray-700 text-sm rounded-full border border-gray-200"
                            whileHover={{
                              scale: 1.05,
                              backgroundColor: "#f9f5ff"
                            }}
                            transition={{ type: "spring", stiffness: 400, damping: 10 }}
                          >
                            {activity}
                          </motion.span>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </motion.div>
      </div>
    </section>
  );
};

export default Volunteering;
