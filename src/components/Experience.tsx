
import React, { useState } from 'react';
import { Button } from './ui/button';
import { Info, ExternalLink } from 'lucide-react';
import { motion } from 'framer-motion';

const experiences = [
  {
    title: "AI & ML Engineer",
    company: "Breyus",
    logo: "/lovable-uploads/Logo.png",
    period: "March 2025 - May 2025",
    description: "Developed Breyus, an AI-powered trade intelligence startup that helps users discover and trade known and lesser-known global commodities. Integrated machine learning, data cleaning, and NLP to predict prices, match buyers and sellers, and extract trade insights from massive, unstructured datasets through a seamless full-stack system.",
    achievements: [
      " AI Price & Delivery Prediction: Trained a Random Forest model to predict commodity prices using region-specific user-submitted data. Helped users estimate realistic trade prices instantly based on trends in the database.",
      " Automated Data Cleaning: Handled highly unstructured, inconsistent product data using pandas and LLM-based interpretation to normalize product descriptions. Removed duplicates, filled nulls, and cleanly separated buyers and sellers from massive datasets.",
      " FastAPI-Based Search Algorithm: Built a lightweight NLP-powered search system that extracts product and country info from queries, matches keywords, and returns top buyer/seller records. Enabled instant discovery of relevant trade leads.",
      " Semantic Tag Generator with spaCy + GloVe: Created an auto-tagging system that analyzes product names using spaCy and word vector similarity. This enriched data labeling even when inputs were vague or unfamiliar.",
      " Product Recommendation Engine: Built a semantic recommendation system that uses word embeddings and cosine similarity to suggest similar products. Enhanced cross-discovery of related commodities during trade exploration.",
      " Full Integration with React Frontend: Hooked all backend AI features into the frontend using FastAPI endpoints without changing the existing UI. Delivered a fluid user experience from search to insight.",
    ],
    image: "/lovable-uploads/Breyus.png",
    details: {
      responsibilities: [
        "Documented all features thoroughly, including pending and completed tasks, and shared structured updates with team members and the founder to ensure clear project tracking and alignment. (Mostly On Whimsical)",
        "Developing and integrating AI backend of Breyus, it seamlessly with the existing React frontend.",
        "Actively participated in brainstorming sessions, idea pitches, and early-stage planning of Breyus to shape it as a globally scalable trade intelligence platform.",
        "Contributed to sprint planning and agile execution, balancing short-term deliverables with long-term system scalability.",
      ],
      technologies: ["Python", "FastAPI", "scikit-learn", "spaCy", "GloVe Embeddings", "Pandas", "NumPy"],
      challenges: "The core challenge was dealing with highly unstructured commodity trade data and building fast, intelligent search and prediction tools from it. We solved this with a custom data cleaning pipeline using pandas and LLMs, keyword-based NLP search with FastAPI, and semantic tagging using spaCy and GloVe.",
      impact: "Enabled global users to discover hidden buyers and sellers of niche commodities with AI assistance. The system provided instant commodity price estimates and intelligent trade insights, accelerating user decision-making and reducing manual research efforts.",
      detailed: [
        "▸ AI Price Prediction System: The AI-powered price prediction system was developed using a data-driven approach where historical price data, contributed by users, was utilized to train machine learning models. This system helps predict the price of various commodities based on regional price data provided by users, allowing both buyers and sellers to estimate fair trade prices instantly. By utilizing Random Forest models, the prediction system accounts for fluctuations in commodity prices across different regions, ensuring accurate and relevant price estimates. The system is crucial in a global trade network, where price transparency and real-time updates are essential for efficient trading.",

        "▸ Automated Data Cleaning Pipeline: The dataset provided for commodity trade was complex, unstructured, and inconsistent, making it a challenge to extract meaningful insights. This feature involved developing an automated data cleaning pipeline using pandas, LLM-based interpretation, and traditional data processing methods. The pipeline performs several tasks such as deduplication, null value imputation, and normalizing product descriptions to ensure consistency. Additionally, it involved categorizing the data into buyers and sellers based on provided details, ensuring a clean dataset for accurate analysis and matching. This feature effectively cleaned lakhs of mixed records, making it easier to search and match buyer and seller data efficiently.",

        "▸ FastAPI-Based Buyer/Seller Search Algorithm: This feature focuses on an intelligent search system built using FastAPI that allows users to find relevant buyers and sellers based on a simple keyword query. By analyzing product names, descriptions, and countries, the system uses basic NLP rules to extract keywords and matches them against a database of buyers and sellers. When users search for a commodity, they are instantly provided with a list of matching buyers and sellers based on product and country relevance. This feature speeds up the process of finding trading opportunities and is vital for increasing trade efficiency and engagement across different regions.",

        "▸ Semantic Tag Generation Using spaCy & GloVe: This feature leverages NLP to automatically generate relevant tags for products based on their descriptions. By using spaCy for tokenizing and analyzing text and GloVe word embeddings for semantic similarity, the system identifies key product features and generates a set of tags that best represent the product. For example, for a product description that mentions 'olive oil,' the system can automatically add tags like 'oil,' 'olive,' and even synonyms like 'vegetable oil.' This enhances the dataset with more comprehensive metadata and allows for easier filtering and discovery of relevant products, even when descriptions are vague or incomplete.",

        "▸ AI-Powered Product Recommendation System: The product recommendation system was designed to improve user experience by suggesting relevant products based on user queries. This system uses GloVe embeddings and spaCy to compute vector representations of product titles and match them with user input. By calculating the cosine similarity between query embeddings and product embeddings, the system recommends top similar products. The more the user interacts with the system, the better the recommendations become, allowing them to discover new commodities or suppliers that best meet their needs. This feature enhances the trading experience, increasing the likelihood of successful trades.",

        "▸ Full Backend Integration with React Frontend: In this feature, all the backend AI-powered functionalities were integrated seamlessly into the frontend application built using React. The integration ensured that users could interact with the platform without disrupting the existing UI or user flows. Using FastAPI endpoints, the React frontend was connected to the backend, enabling real-time search results, price predictions, and product recommendations. This end-to-end integration provided a smooth and interactive user experience, making it easy for users to access AI-driven insights directly through the application. By maintaining the existing UI structure, the integration ensured that the system's functionality was enhanced without requiring a major redesign.",
      ],
    }
  }
];

const virtualInternships = [
  {
    title: "Data Science Job Simulation",
    company: "British Airways",
    period: "2024",
    description: "Completed a self-paced virtual internship simulating real-world data science challenges at British Airways, focusing on customer insights and predictive analytics.",
    achievements: [
      "Scraped and analyzed customer review data to extract actionable insights using Python and NLTK",
      "Performed sentiment analysis to classify reviews as positive, neutral, or negative",
      "Developed a classification model to predict customer buying behavior",
      "Presented findings through PowerPoint to inform business strategies"
    ],
    technologies: ["Python", "Pandas", "scikit-learn", "NLTK", "Web Scraping", "Sentiment Analysis", "Machine Learning", "Data Visualization"],
    certificateUrl: "https://drive.google.com/file/d/1MGGuR0ieLKLvl1WvqIlmOI1A6U7yLoR_/view?usp=sharing",
    logo: "/lovable-uploads/06c5e679-61c5-492d-b201-8e8d2163854c.png"
  },
  {
    title: "Data Science Job Simulation",
    company: "BCG X",
    period: "2025",
    description: "Completed a self-paced virtual internship simulating real-world data science challenges at BCG X, focusing on customer churn analysis and predictive modeling for a utility company.",
    achievements: [
      "Formulated business hypotheses to identify drivers of customer churn",
      "Conducted exploratory data analysis to uncover patterns and correlations",
      "Engineered features and developed a Random Forest classifier to predict churn",
      "Presented findings and strategic recommendations to stakeholders"
    ],
    technologies: ["Python", "Pandas", "scikit-learn", "Feature Engineering", "Exploratory Data Analysis", "Machine Learning", "Data Visualization"],
    certificateUrl: "https://drive.google.com/file/d/1U2mCs94yXXDbtB3uxUPSFBZfcHalNqex/view?usp=sharing",
    logo: "/lovable-uploads/d9eda68a-8256-4604-ac47-adbf1b30868b.png"
  }
];

const Experience = () => {
  const [expandedIndex, setExpandedIndex] = useState(null);
  const [expandedVirtualIndex, setExpandedVirtualIndex] = useState(null);

  const toggleExpand = (index) => {
    if (expandedIndex === index) {
      setExpandedIndex(null);
    } else {
      setExpandedIndex(index);
    }
  };

  const toggleVirtualExpand = (index) => {
    if (expandedVirtualIndex === index) {
      setExpandedVirtualIndex(null);
    } else {
      setExpandedVirtualIndex(index);
    }
  };

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
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.6,
        ease: "easeOut"
      }
    }
  };

  const detailsVariants = {
    hidden: { opacity: 0, height: 0 },
    visible: {
      opacity: 1,
      height: "auto",
      transition: {
        duration: 0.5,
        ease: "easeInOut"
      }
    }
  };

  return (
    <section className="py-20 px-6 bg-gradient-to-b from-gray-50 to-white">
      <div className="max-w-6xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="text-4xl font-bold text-center mb-2 text-gray-800 bg-clip-text text-transparent bg-gradient-to-r from-gray-800 to-brand-purple"
        >
          Real-World AI Work
        </motion.h2>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.2 }}
          className="text-gray-600 text-center mb-12 text-lg"
        >
          Where I've applied my skills to solve real business problems.
        </motion.p>

        <motion.div
          className="relative"
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
        >
          {experiences.map((exp, index) => (
            <motion.div
              key={index}
              className="timeline-item"
              variants={itemVariants}
            >
              <motion.div
                className="mb-8 bg-white rounded-xl p-6 shadow-md hover:shadow-lg transition-all duration-300"
                whileHover={{ y: -5 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="text-2xl font-bold text-gray-800 bg-clip-text text-transparent bg-gradient-to-r from-gray-800 to-brand-purple">{exp.title}</h3>
                    <p className="text-gray-600">{exp.company} · {exp.period}</p>
                  </div>
                  <div className="w-14 h-14 rounded-full bg-gradient-to-br text-white flex items-center justify-center font-bold text-2xl shadow-lg">
                    <img src={exp.logo} alt="Company logo" className="w-full h-full object-cover rounded-full" />
                  </div>
                </div>

                <p className="mt-4 text-gray-700">{exp.description}</p>

                <div className="mt-4 p-4 bg-green-50 rounded-md border border-green-100 text-green-700">
                  {exp.achievements.map((achievement, i) => (
                    <div key={i} className="flex items-start mb-2 last:mb-0">
                      <span className="mr-2">▸</span>
                      <span>{achievement}</span>
                    </div>
                  ))}
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  className="mt-4 flex items-center gap-1 rounded-full border-none bg-purple-100 text-black font-semibold hover:bg-purple-200 transition-colors"
                  onClick={() => toggleExpand(index)}
                >
                  <Info size={16} />
                  {expandedIndex === index ? "Hide Details" : "View Details"}
                </Button>

                <motion.div
                  variants={detailsVariants}
                  initial="hidden"
                  animate={expandedIndex === index ? "visible" : "hidden"}
                  className="overflow-hidden"
                >
                  <div className="mt-6 pt-6 border-t border-gray-100">
                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <div className="mb-6 overflow-hidden rounded-lg shadow-md">
                          <img
                            src={exp.image}
                            alt={`${exp.company} project visualization`}
                            className="w-full h-48 object-cover hover:scale-105 transition-transform duration-500"
                          />
                        </div>
                        <h4 className="font-semibold text-lg mb-3 text-gray-800">Responsibilities</h4>
                        <ul className="list-disc pl-5 space-y-1 text-gray-700 mb-6">
                          {exp.details.responsibilities.map((item, i) => (
                            <li key={i}>{item}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h4 className="font-semibold text-lg mb-3 text-gray-800">Technologies Used</h4>
                        <div className="flex flex-wrap gap-2 mb-6">
                          {exp.details.technologies.map((tech, i) => (
                            <span key={i} className="px-3 py-1.5 bg-purple-200 text-black text-sm font-bold rounded-full transition-all hover:shadow-md border border-purple-300">
                              {tech}
                            </span>
                          ))}
                        </div>

                        <h4 className="font-semibold text-lg mb-3 text-gray-800">Challenges & Solutions</h4>
                        <p className="text-gray-700 mb-6">{exp.details.challenges}</p>

                        <h4 className="font-semibold text-lg mb-3 text-gray-800">Business Impact</h4>
                        <p className="text-gray-700">{exp.details.impact}</p>
                      </div>

                      <div className="col-span-full mt-8 bg-gradient-to-r from-gray-50 to-white rounded-lg p-6 border border-gray-100">
                        <h4 className="font-bold text-xl mb-6 text-center text-gray-800 bg-clip-text text-transparent bg-gradient-to-r from-gray-800 to-brand-purple">
                          📋 All Features Explained in Detail
                        </h4>
                        <div className="space-y-6">
                          {exp.details.detailed.map((detail, i) => (
                            <div key={i} className="bg-white rounded-lg p-5 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                              <p className="text-gray-700 leading-relaxed text-justify">
                                {detail}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              </motion.div>
            </motion.div>
          ))}
        </motion.div>

        {/* Virtual Internships Section */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          viewport={{ once: true }}
          className="mt-16"
        >
          <h3 className="text-3xl font-bold text-center mb-8 text-gray-800 bg-clip-text text-transparent bg-gradient-to-r from-gray-800 to-brand-purple">
            Virtual Internships
          </h3>
          <p className="text-gray-600 text-center mb-12 text-lg">
            Virtual learning experiences that shaped my technical foundation.
          </p>

          <motion.div
            className="space-y-6"
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
          >
            {virtualInternships.map((internship, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                className="bg-white rounded-xl p-6 shadow-md hover:shadow-lg transition-all duration-300"
                whileHover={{ y: -3 }}
              >
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h4 className="text-xl font-bold text-gray-800">{internship.title}</h4>
                    <p className="text-gray-600">{internship.company} · {internship.period}</p>
                  </div>
                  <div className="w-12 h-12 rounded-full bg-white border-2 border-gray-200 flex items-center justify-center shadow-lg">
                    <img src={internship.logo} alt={`${internship.company} logo`} className="w-8 h-8 object-contain" />
                  </div>
                </div>

                <p className="text-gray-700 mb-4">{internship.description}</p>

                <div className="mb-4 p-4 bg-blue-50 rounded-md border border-blue-100 text-blue-700">
                  {internship.achievements.map((achievement, i) => (
                    <div key={i} className="flex items-start mb-2 last:mb-0">
                      <span className="mr-2">▸</span>
                      <span>{achievement}</span>
                    </div>
                  ))}
                </div>

                <div className="flex flex-wrap gap-2 mb-4">
                  {internship.technologies.map((tech, i) => (
                    <span key={i} className="px-3 py-1 bg-gradient-to-r from-blue-100 to-purple-100 text-blue-700 text-sm rounded-full">
                      {tech}
                    </span>
                  ))}
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  className="flex items-center gap-2 rounded-full border-blue-500 text-blue-600 hover:bg-blue-50"
                  onClick={() => window.open(internship.certificateUrl, '_blank')}
                >
                  <ExternalLink size={16} />
                  View Certificate
                </Button>
              </motion.div>
            ))}
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};

export default Experience;
