import React, { useState } from 'react';
import { 
  XMarkIcon, 
  HeartIcon, 
  ShareIcon, 
  PencilIcon, 
  TrashIcon,
  CalendarIcon,
  MapPinIcon,
  TagIcon,
  SparklesIcon,
  PhotoIcon,
  PlusIcon,
  EyeIcon
} from '@heroicons/react/24/outline';
import { HeartIcon as HeartSolidIcon } from '@heroicons/react/24/solid';
import { ITEM_CATEGORIES, RARITY_LEVELS, EMOTIONS } from '../api/userItemsApi';

const ItemDetailModal = ({ 
  item, 
  isOpen, 
  onClose, 
  onEdit, 
  onDelete, 
  onToggleFavorite,
  onAddMemory,
  onShare 
}) => {
  const [activeTab, setActiveTab] = useState('details');
  const [newMemory, setNewMemory] = useState({
    title: '',
    description: '',
    emotion: 'happy'
  });
  const [isAddingMemory, setIsAddingMemory] = useState(false);
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);

  if (!isOpen || !item) return null;

  const getCategoryInfo = (category) => {
    return ITEM_CATEGORIES.find(cat => cat.value === category) || 
           { label: category, icon: '📦' };
  };

  const getRarityInfo = (rarity) => {
    return RARITY_LEVELS.find(r => r.value === rarity) || 
           RARITY_LEVELS[0];
  };

  const getEmotionInfo = (emotion) => {
    return EMOTIONS.find(e => e.value === emotion) || 
           { label: emotion, icon: '😊' };
  };

  const formatDate = (date) => {
    return new Date(date).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleAddMemory = async () => {
    if (!newMemory.title.trim()) return;
    
    setIsAddingMemory(true);
    try {
      await onAddMemory(item._id, newMemory);
      setNewMemory({ title: '', description: '', emotion: 'happy' });
    } catch (error) {
      console.error('Error adding memory:', error);
    } finally {
      setIsAddingMemory(false);
    }
  };

  const categoryInfo = getCategoryInfo(item.category);
  const rarityInfo = getRarityInfo(item.rarity);
  const images = item.media?.images || [];
  const memories = item.metadata?.memories || [];
  const tags = item.metadata?.tags || [];

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900/95 backdrop-blur-md rounded-2xl border border-white/20 w-full max-w-4xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-white/10">
          <div className="flex items-center gap-4">
            <div className="text-3xl">{categoryInfo.icon}</div>
            <div>
              <h2 className="text-2xl font-bold text-white">{item.name}</h2>
              <div className="flex items-center gap-2 text-sm text-gray-300">
                <span>{categoryInfo.label}</span>
                <span>•</span>
                <span className="capitalize">{item.type}</span>
                <span>•</span>
                <span className={`${rarityInfo.color} font-medium`}>
                  <SparklesIcon className="w-4 h-4 inline mr-1" />
                  {rarityInfo.label}
                </span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {/* Action Buttons */}
            <button
              onClick={() => onToggleFavorite(item._id)}
              className={`
                p-2 rounded-full transition-all duration-200
                ${item.metadata?.isFavorite 
                  ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30' 
                  : 'bg-white/10 text-white/60 hover:bg-white/20 hover:text-white'
                }
              `}
              title={item.metadata?.isFavorite ? 'Remove from favorites' : 'Add to favorites'}
            >
              {item.metadata?.isFavorite ? (
                <HeartSolidIcon className="w-5 h-5" />
              ) : (
                <HeartIcon className="w-5 h-5" />
              )}
            </button>
            
            <button
              onClick={() => onEdit(item)}
              className="p-2 bg-white/10 text-white/60 hover:bg-white/20 hover:text-white rounded-full transition-colors"
              title="Edit item"
            >
              <PencilIcon className="w-5 h-5" />
            </button>
            
            <button
              onClick={() => onShare(item)}
              className="p-2 bg-white/10 text-white/60 hover:bg-white/20 hover:text-white rounded-full transition-colors"
              title="Share item"
            >
              <ShareIcon className="w-5 h-5" />
            </button>
            
            <button
              onClick={() => onDelete(item)}
              className="p-2 bg-red-500/20 text-red-400 hover:bg-red-500/30 rounded-full transition-colors"
              title="Delete item"
            >
              <TrashIcon className="w-5 h-5" />
            </button>
            
            <button
              onClick={onClose}
              className="p-2 text-gray-400 hover:text-white transition-colors ml-2"
            >
              <XMarkIcon className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex flex-col lg:flex-row h-[calc(90vh-120px)]">
          {/* Left Side - Image Gallery */}
          <div className="lg:w-1/2 p-6 border-r border-white/10">
            {images.length > 0 ? (
              <div className="space-y-4">
                {/* Main Image */}
                <div className="aspect-square bg-gray-800 rounded-xl overflow-hidden">
                  <img 
                    src={images[selectedImageIndex]?.url} 
                    alt={item.name}
                    className="w-full h-full object-cover"
                  />
                </div>
                
                {/* Image Thumbnails */}
                {images.length > 1 && (
                  <div className="flex gap-2 overflow-x-auto">
                    {images.map((image, index) => (
                      <button
                        key={index}
                        onClick={() => setSelectedImageIndex(index)}
                        className={`
                          flex-shrink-0 w-16 h-16 rounded-lg overflow-hidden border-2 transition-colors
                          ${index === selectedImageIndex ? 'border-purple-500' : 'border-white/20'}
                        `}
                      >
                        <img 
                          src={image.url} 
                          alt={`${item.name} ${index + 1}`}
                          className="w-full h-full object-cover"
                        />
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="aspect-square bg-gray-800 rounded-xl flex items-center justify-center">
                <div className="text-center text-gray-400">
                  <PhotoIcon className="w-16 h-16 mx-auto mb-4" />
                  <p>No images available</p>
                </div>
              </div>
            )}
          </div>

          {/* Right Side - Details */}
          <div className="lg:w-1/2 flex flex-col">
            {/* Tabs */}
            <div className="flex border-b border-white/10">
              {[
                { id: 'details', label: 'Details', icon: EyeIcon },
                { id: 'memories', label: 'Memories', icon: HeartIcon, count: memories.length }
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`
                    flex items-center gap-2 px-6 py-4 text-sm font-medium transition-colors
                    ${activeTab === tab.id 
                      ? 'text-purple-400 border-b-2 border-purple-400' 
                      : 'text-gray-400 hover:text-white'
                    }
                  `}
                >
                  <tab.icon className="w-4 h-4" />
                  {tab.label}
                  {tab.count !== undefined && (
                    <span className="bg-white/10 text-xs px-2 py-1 rounded-full">
                      {tab.count}
                    </span>
                  )}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto p-6">
              {activeTab === 'details' && (
                <div className="space-y-6">
                  {/* Description */}
                  {item.description && (
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-2">Description</h3>
                      <p className="text-gray-300 leading-relaxed">{item.description}</p>
                    </div>
                  )}

                  {/* Properties */}
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-3">Properties</h3>
                    <div className="grid grid-cols-2 gap-4">
                      {item.properties?.color && (
                        <div>
                          <span className="text-sm text-gray-400">Color</span>
                          <p className="text-white">{item.properties.color}</p>
                        </div>
                      )}
                      {item.properties?.size && (
                        <div>
                          <span className="text-sm text-gray-400">Size</span>
                          <p className="text-white">{item.properties.size}</p>
                        </div>
                      )}
                      {item.properties?.material && (
                        <div>
                          <span className="text-sm text-gray-400">Material</span>
                          <p className="text-white">{item.properties.material}</p>
                        </div>
                      )}
                      {item.properties?.condition && (
                        <div>
                          <span className="text-sm text-gray-400">Condition</span>
                          <p className="text-white capitalize">{item.properties.condition}</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Value */}
                  {(item.value?.points > 0 || item.value?.coins > 0 || item.value?.monetaryValue > 0) && (
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-3">Value</h3>
                      <div className="flex gap-4">
                        {item.value.points > 0 && (
                          <div className="text-center">
                            <div className="text-2xl font-bold text-yellow-400">{item.value.points}</div>
                            <div className="text-sm text-gray-400">Points</div>
                          </div>
                        )}
                        {item.value.coins > 0 && (
                          <div className="text-center">
                            <div className="text-2xl font-bold text-blue-400">{item.value.coins}</div>
                            <div className="text-sm text-gray-400">Coins</div>
                          </div>
                        )}
                        {item.value.monetaryValue > 0 && (
                          <div className="text-center">
                            <div className="text-2xl font-bold text-green-400">
                              ${item.value.monetaryValue.toFixed(2)}
                            </div>
                            <div className="text-sm text-gray-400">{item.value.currency || 'USD'}</div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Source & Location */}
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-3">Source & Location</h3>
                    <div className="space-y-3">
                      <div className="flex items-center gap-2 text-gray-300">
                        <span className="text-sm text-gray-400">Source:</span>
                        <span className="capitalize">{item.location?.source || 'Unknown'}</span>
                        {item.location?.sourceName && (
                          <>
                            <span>•</span>
                            <span>{item.location.sourceName}</span>
                          </>
                        )}
                      </div>
                      
                      {item.location?.acquiredAt && (
                        <div className="flex items-center gap-2 text-gray-300">
                          <CalendarIcon className="w-4 h-4 text-gray-400" />
                          <span>Acquired on {formatDate(item.location.acquiredAt)}</span>
                        </div>
                      )}
                      
                      {item.location?.acquiredLocation?.address && (
                        <div className="flex items-center gap-2 text-gray-300">
                          <MapPinIcon className="w-4 h-4 text-gray-400" />
                          <span>{item.location.acquiredLocation.address}</span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Tags */}
                  {tags.length > 0 && (
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-3">Tags</h3>
                      <div className="flex flex-wrap gap-2">
                        {tags.map((tag, index) => (
                          <span 
                            key={index}
                            className="inline-flex items-center gap-1 px-3 py-1 bg-white/10 rounded-full text-sm text-gray-300"
                          >
                            <TagIcon className="w-3 h-3" />
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Notes */}
                  {item.metadata?.notes && (
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-2">Notes</h3>
                      <p className="text-gray-300 leading-relaxed">{item.metadata.notes}</p>
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'memories' && (
                <div className="space-y-6">
                  {/* Add Memory Form */}
                  <div className="bg-white/5 rounded-xl p-4 space-y-4">
                    <h3 className="text-lg font-semibold text-white">Add Memory</h3>
                    
                    <div>
                      <input
                        type="text"
                        value={newMemory.title}
                        onChange={(e) => setNewMemory(prev => ({ ...prev, title: e.target.value }))}
                        className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                        placeholder="Memory title..."
                      />
                    </div>
                    
                    <div>
                      <textarea
                        value={newMemory.description}
                        onChange={(e) => setNewMemory(prev => ({ ...prev, description: e.target.value }))}
                        rows={3}
                        className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                        placeholder="Describe your memory..."
                      />
                    </div>
                    
                    <div className="flex gap-3">
                      <select
                        value={newMemory.emotion}
                        onChange={(e) => setNewMemory(prev => ({ ...prev, emotion: e.target.value }))}
                        className="flex-1 px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                      >
                        {EMOTIONS.map(emotion => (
                          <option key={emotion.value} value={emotion.value} className="bg-gray-800">
                            {emotion.icon} {emotion.label}
                          </option>
                        ))}
                      </select>
                      
                      <button
                        onClick={handleAddMemory}
                        disabled={!newMemory.title.trim() || isAddingMemory}
                        className="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
                      >
                        {isAddingMemory ? 'Adding...' : 'Add'}
                      </button>
                    </div>
                  </div>

                  {/* Memories List */}
                  <div className="space-y-4">
                    {memories.length > 0 ? (
                      memories.map((memory, index) => {
                        const emotionInfo = getEmotionInfo(memory.emotion);
                        return (
                          <div key={index} className="bg-white/5 rounded-xl p-4 space-y-2">
                            <div className="flex items-center justify-between">
                              <h4 className="font-semibold text-white">{memory.title}</h4>
                              <div className="flex items-center gap-2 text-sm text-gray-400">
                                <span>{emotionInfo.icon}</span>
                                <span>{emotionInfo.label}</span>
                                {memory.date && (
                                  <>
                                    <span>•</span>
                                    <span>{formatDate(memory.date)}</span>
                                  </>
                                )}
                              </div>
                            </div>
                            {memory.description && (
                              <p className="text-gray-300 text-sm leading-relaxed">
                                {memory.description}
                              </p>
                            )}
                          </div>
                        );
                      })
                    ) : (
                      <div className="text-center py-8 text-gray-400">
                        <HeartIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>No memories yet. Add your first memory above!</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ItemDetailModal;
