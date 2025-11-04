import React, { useState } from 'react';
import { 
  XMarkIcon, 
  PhotoIcon, 
  PlusIcon, 
  MapPinIcon,
  TagIcon,
  SparklesIcon
} from '@heroicons/react/24/outline';
import { ITEM_CATEGORIES, ITEM_TYPES, RARITY_LEVELS, EMOTIONS } from '../api/userItemsApi';

const AddItemModal = ({ isOpen, onClose, onSubmit, initialData = null }) => {
  const [formData, setFormData] = useState({
    name: initialData?.name || '',
    description: initialData?.description || '',
    category: initialData?.category || 'collectible',
    type: initialData?.type || 'physical',
    rarity: initialData?.rarity || 'common',
    value: {
      points: initialData?.value?.points || 0,
      coins: initialData?.value?.coins || 0,
      monetaryValue: initialData?.value?.monetaryValue || 0,
      currency: initialData?.value?.currency || 'USD'
    },
    properties: {
      color: initialData?.properties?.color || '',
      size: initialData?.properties?.size || '',
      material: initialData?.properties?.material || '',
      brand: initialData?.properties?.brand || '',
      condition: initialData?.properties?.condition || 'new'
    },
    source: initialData?.source || 'other',
    sourceName: initialData?.sourceName || '',
    acquiredLocation: {
      address: initialData?.acquiredLocation?.address || '',
      venue: initialData?.acquiredLocation?.venue || ''
    },
    tags: initialData?.tags?.join(', ') || '',
    isPublic: initialData?.isPublic || false,
    isFavorite: initialData?.isFavorite || false,
    notes: initialData?.notes || '',
    images: []
  });

  const [currentTag, setCurrentTag] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errors, setErrors] = useState({});

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: null }));
    }
  };

  const handleNestedInputChange = (parent, field, value) => {
    setFormData(prev => ({
      ...prev,
      [parent]: {
        ...prev[parent],
        [field]: value
      }
    }));
  };

  const addTag = () => {
    if (currentTag.trim() && !formData.tags.includes(currentTag.trim())) {
      const newTags = formData.tags ? `${formData.tags}, ${currentTag.trim()}` : currentTag.trim();
      setFormData(prev => ({ ...prev, tags: newTags }));
      setCurrentTag('');
    }
  };

  const removeTag = (tagToRemove) => {
    const tagsArray = formData.tags.split(',').map(tag => tag.trim()).filter(Boolean);
    const updatedTags = tagsArray.filter(tag => tag !== tagToRemove).join(', ');
    setFormData(prev => ({ ...prev, tags: updatedTags }));
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.name.trim()) {
      newErrors.name = 'Item name is required';
    }
    
    if (!formData.category) {
      newErrors.category = 'Category is required';
    }
    
    if (!formData.type) {
      newErrors.type = 'Type is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);
    
    try {
      const submitData = {
        ...formData,
        tags: formData.tags.split(',').map(tag => tag.trim()).filter(Boolean),
        metadata: {
          tags: formData.tags.split(',').map(tag => tag.trim()).filter(Boolean),
          isPublic: formData.isPublic,
          isFavorite: formData.isFavorite,
          notes: formData.notes
        }
      };

      await onSubmit(submitData);
      onClose();
    } catch (error) {
      console.error('Error submitting item:', error);
      setErrors({ submit: error.message || 'Failed to save item' });
    } finally {
      setIsSubmitting(false);
    }
  };

  const getCategoryIcon = (category) => {
    const categoryInfo = ITEM_CATEGORIES.find(cat => cat.value === category);
    return categoryInfo?.icon || '📦';
  };

  const getRarityColor = (rarity) => {
    const rarityInfo = RARITY_LEVELS.find(r => r.value === rarity);
    return rarityInfo?.color || 'text-gray-400';
  };

  if (!isOpen) return null;

  const tagsArray = formData.tags.split(',').map(tag => tag.trim()).filter(Boolean);

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900/95 backdrop-blur-md rounded-2xl border border-white/20 w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-white/10">
          <h2 className="text-2xl font-bold text-white">
            {initialData ? 'Edit Item' : 'Add New Item'}
          </h2>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-white transition-colors"
          >
            <XMarkIcon className="w-6 h-6" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Basic Info */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">Basic Information</h3>
            
            {/* Name */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Item Name *
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => handleInputChange('name', e.target.value)}
                className={`
                  w-full px-4 py-3 bg-white/10 border rounded-lg text-white placeholder-gray-400
                  focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors
                  ${errors.name ? 'border-red-500' : 'border-white/20'}
                `}
                placeholder="Enter item name..."
              />
              {errors.name && (
                <p className="text-red-400 text-sm mt-1">{errors.name}</p>
              )}
            </div>

            {/* Description */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Description
              </label>
              <textarea
                value={formData.description}
                onChange={(e) => handleInputChange('description', e.target.value)}
                rows={3}
                className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors resize-none"
                placeholder="Describe your item..."
              />
            </div>

            {/* Category and Type */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Category *
                </label>
                <select
                  value={formData.category}
                  onChange={(e) => handleInputChange('category', e.target.value)}
                  className={`
                    w-full px-4 py-3 bg-white/10 border rounded-lg text-white
                    focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors
                    ${errors.category ? 'border-red-500' : 'border-white/20'}
                  `}
                >
                  {ITEM_CATEGORIES.map(category => (
                    <option key={category.value} value={category.value} className="bg-gray-800">
                      {category.icon} {category.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Type *
                </label>
                <select
                  value={formData.type}
                  onChange={(e) => handleInputChange('type', e.target.value)}
                  className={`
                    w-full px-4 py-3 bg-white/10 border rounded-lg text-white
                    focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors
                    ${errors.type ? 'border-red-500' : 'border-white/20'}
                  `}
                >
                  {ITEM_TYPES.map(type => (
                    <option key={type.value} value={type.value} className="bg-gray-800">
                      {type.icon} {type.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Rarity */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Rarity
              </label>
              <select
                value={formData.rarity}
                onChange={(e) => handleInputChange('rarity', e.target.value)}
                className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors"
              >
                {RARITY_LEVELS.map(rarity => (
                  <option key={rarity.value} value={rarity.value} className="bg-gray-800">
                    {rarity.label}
                  </option>
                ))}
              </select>
              <div className="mt-2 flex items-center gap-2">
                <SparklesIcon className="w-4 h-4 text-gray-400" />
                <span className={`text-sm ${getRarityColor(formData.rarity)}`}>
                  {RARITY_LEVELS.find(r => r.value === formData.rarity)?.label} Item
                </span>
              </div>
            </div>
          </div>

          {/* Value */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">Value</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Points
                </label>
                <input
                  type="number"
                  min="0"
                  value={formData.value.points}
                  onChange={(e) => handleNestedInputChange('value', 'points', parseInt(e.target.value) || 0)}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors"
                  placeholder="0"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Coins
                </label>
                <input
                  type="number"
                  min="0"
                  value={formData.value.coins}
                  onChange={(e) => handleNestedInputChange('value', 'coins', parseInt(e.target.value) || 0)}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors"
                  placeholder="0"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Monetary Value
                </label>
                <input
                  type="number"
                  min="0"
                  step="0.01"
                  value={formData.value.monetaryValue}
                  onChange={(e) => handleNestedInputChange('value', 'monetaryValue', parseFloat(e.target.value) || 0)}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors"
                  placeholder="0.00"
                />
              </div>
            </div>
          </div>

          {/* Properties */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">Properties</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Color
                </label>
                <input
                  type="text"
                  value={formData.properties.color}
                  onChange={(e) => handleNestedInputChange('properties', 'color', e.target.value)}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors"
                  placeholder="e.g., Blue, Red..."
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Size
                </label>
                <input
                  type="text"
                  value={formData.properties.size}
                  onChange={(e) => handleNestedInputChange('properties', 'size', e.target.value)}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors"
                  placeholder="e.g., Small, Large..."
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Material
                </label>
                <input
                  type="text"
                  value={formData.properties.material}
                  onChange={(e) => handleNestedInputChange('properties', 'material', e.target.value)}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors"
                  placeholder="e.g., Cotton, Metal..."
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Condition
                </label>
                <select
                  value={formData.properties.condition}
                  onChange={(e) => handleNestedInputChange('properties', 'condition', e.target.value)}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors"
                >
                  <option value="new" className="bg-gray-800">New</option>
                  <option value="excellent" className="bg-gray-800">Excellent</option>
                  <option value="good" className="bg-gray-800">Good</option>
                  <option value="fair" className="bg-gray-800">Fair</option>
                  <option value="poor" className="bg-gray-800">Poor</option>
                </select>
              </div>
            </div>
          </div>

          {/* Source and Location */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">Source & Location</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Source
                </label>
                <select
                  value={formData.source}
                  onChange={(e) => handleInputChange('source', e.target.value)}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors"
                >
                  <option value="adventure" className="bg-gray-800">Adventure</option>
                  <option value="event" className="bg-gray-800">Event</option>
                  <option value="purchase" className="bg-gray-800">Purchase</option>
                  <option value="gift" className="bg-gray-800">Gift</option>
                  <option value="achievement" className="bg-gray-800">Achievement</option>
                  <option value="reward" className="bg-gray-800">Reward</option>
                  <option value="other" className="bg-gray-800">Other</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Source Name
                </label>
                <input
                  type="text"
                  value={formData.sourceName}
                  onChange={(e) => handleInputChange('sourceName', e.target.value)}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors"
                  placeholder="e.g., Downtown Food Tour"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  <MapPinIcon className="w-4 h-4 inline mr-1" />
                  Location
                </label>
                <input
                  type="text"
                  value={formData.acquiredLocation.address}
                  onChange={(e) => handleNestedInputChange('acquiredLocation', 'address', e.target.value)}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors"
                  placeholder="Address or location"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Venue
                </label>
                <input
                  type="text"
                  value={formData.acquiredLocation.venue}
                  onChange={(e) => handleNestedInputChange('acquiredLocation', 'venue', e.target.value)}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors"
                  placeholder="Venue or business name"
                />
              </div>
            </div>
          </div>

          {/* Tags */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">Tags</h3>
            <div className="space-y-3">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={currentTag}
                  onChange={(e) => setCurrentTag(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
                  className="flex-1 px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors"
                  placeholder="Add a tag..."
                />
                <button
                  type="button"
                  onClick={addTag}
                  className="px-4 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
                >
                  <PlusIcon className="w-5 h-5" />
                </button>
              </div>
              
              {tagsArray.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {tagsArray.map((tag, index) => (
                    <span
                      key={index}
                      className="inline-flex items-center gap-1 px-3 py-1 bg-white/10 rounded-full text-sm text-gray-300"
                    >
                      <TagIcon className="w-3 h-3" />
                      {tag}
                      <button
                        type="button"
                        onClick={() => removeTag(tag)}
                        className="ml-1 text-gray-400 hover:text-white"
                      >
                        <XMarkIcon className="w-3 h-3" />
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Notes */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Notes
            </label>
            <textarea
              value={formData.notes}
              onChange={(e) => handleInputChange('notes', e.target.value)}
              rows={3}
              className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors resize-none"
              placeholder="Personal notes about this item..."
            />
          </div>

          {/* Settings */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">Settings</h3>
            <div className="space-y-3">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={formData.isFavorite}
                  onChange={(e) => handleInputChange('isFavorite', e.target.checked)}
                  className="w-5 h-5 text-purple-600 bg-white/10 border-white/20 rounded focus:ring-purple-500"
                />
                <span className="text-gray-300">Mark as favorite</span>
              </label>
              
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={formData.isPublic}
                  onChange={(e) => handleInputChange('isPublic', e.target.checked)}
                  className="w-5 h-5 text-purple-600 bg-white/10 border-white/20 rounded focus:ring-purple-500"
                />
                <span className="text-gray-300">Make public (visible to others)</span>
              </label>
            </div>
          </div>

          {/* Error Message */}
          {errors.submit && (
            <div className="p-4 bg-red-500/20 border border-red-500/50 rounded-lg">
              <p className="text-red-400 text-sm">{errors.submit}</p>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-3 pt-6 border-t border-white/10">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-6 py-3 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="flex-1 px-6 py-3 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
            >
              {isSubmitting ? 'Saving...' : (initialData ? 'Update Item' : 'Add Item')}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AddItemModal;
